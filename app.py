from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import uuid
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
CHECKPOINT_PATH = '/content/drive/MyDrive/EchoMind/model_best.pt'
MAX_AUDIO_LENGTH = 5 * 16000  # 5 seconds at 16kHz
SAMPLE_RATE = 16000

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

#############################################
#  MODEL CLASSES
#############################################

class CrossAttention(nn.Module):
    def __init__(self, audio_dim, text_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (audio_dim // n_heads) ** -0.5
        self.to_q = nn.Linear(audio_dim, audio_dim)
        self.to_k = nn.Linear(text_dim, audio_dim)
        self.to_v = nn.Linear(text_dim, audio_dim)
        self.to_out = nn.Linear(audio_dim, audio_dim)
    
    def forward(self, x, context):
        B, C, T = x.shape
        x_flat = rearrange(x, 'b c t -> b t c')
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.n_heads)
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhqk,bhvd->bhqd', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.to_out(out)
        return rearrange(out, 'b t c -> b c t')

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x + residual

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c, text_dim=768, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.conv = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.res1 = ResidualBlock(out_c)
        self.res2 = ResidualBlock(out_c)
        if use_attn:
            self.attn = CrossAttention(out_c, text_dim)
        self.downsample = nn.Conv1d(out_c, out_c, 4, stride=2, padding=1)
    
    def forward(self, x, text_emb=None):
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        if self.use_attn and text_emb is not None:
            x = x + self.attn(x, text_emb)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c, text_dim=768, use_attn=False):
        super().__init__()
        self.use_attn = use_attn
        self.upsample = nn.ConvTranspose1d(in_c, out_c, 4, stride=2, padding=1)
        self.conv = nn.Conv1d(out_c + skip_c, out_c, 3, padding=1)
        self.res1 = ResidualBlock(out_c)
        self.res2 = ResidualBlock(out_c)
        if use_attn:
            self.attn = CrossAttention(out_c, text_dim)
    
    def forward(self, x, skip, text_emb=None):
        x = self.upsample(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        if self.use_attn and text_emb is not None:
            x = x + self.attn(x, text_emb)
        return x

class AdvancedUNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 256], text_dim=768):
        super().__init__()
        self.input_conv = nn.Conv1d(1, channels[0], 7, padding=3)
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            use_attn = i >= 2
            self.down_blocks.append(DownBlock(channels[i], channels[i+1], channels[i], text_dim, use_attn))
        self.mid_block1 = ResidualBlock(channels[-1])
        self.mid_attn = CrossAttention(channels[-1], text_dim)
        self.mid_block2 = ResidualBlock(channels[-1])
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            use_attn = i >= 2
            skip_c = channels[i]
            self.up_blocks.append(UpBlock(channels[i], channels[i-1], skip_c, text_dim, use_attn))
        self.output_conv = nn.Conv1d(channels[0], 1, 7, padding=3)
    
    def forward(self, x, text_emb):
        original_length = x.size(-1)
        x = self.input_conv(x)
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, text_emb)
            skips.append(skip)
        x = self.mid_block1(x)
        x = x + self.mid_attn(x, text_emb)
        x = self.mid_block2(x)
        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip, text_emb)
        x = self.output_conv(x)
        if x.size(-1) != original_length:
            x = F.interpolate(x, size=original_length, mode='linear', align_corners=False)
        return x

class FullModel(nn.Module):
    def __init__(self, unet_channels=[32, 64, 128, 256, 256], text_dim=768):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.unet = AdvancedUNet(unet_channels, text_dim)
    
    def forward(self, wav, input_ids, attention_mask):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_output.last_hidden_state
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        return self.unet(wav, text_emb)

#############################################
#  LOAD MODEL ON STARTUP
#############################################

print("🔄 Loading model and tokenizer...")
try:
    model = FullModel().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"✅ Model loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    tokenizer = None

#############################################
#  INFERENCE FUNCTION
#############################################

def process_audio(audio_path, prompt):
    """
    Process audio with text prompt
    Returns: (output_path, metadata) or (None, error_message)
    """
    try:
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        original_duration = wav.size(1) / sr
        
        # Resample to 16kHz
        if sr != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        
        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Limit to 5 seconds
        was_truncated = False
        if wav.size(1) > MAX_AUDIO_LENGTH:
            wav = wav[:, :MAX_AUDIO_LENGTH]
            was_truncated = True
        
        # Tokenize prompt
        tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Move to device
        wav = wav.unsqueeze(0).to(device)
        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)
        
        # Generate
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                output = model(wav, ids, mask)
        
        # Save output
        output = output.squeeze(0).cpu()
        output_filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        torchaudio.save(output_path, output, sample_rate=SAMPLE_RATE)
        
        metadata = {
            'original_duration': round(original_duration, 2),
            'output_duration': round(output.size(1) / SAMPLE_RATE, 2),
            'was_truncated': was_truncated,
            'sample_rate': SAMPLE_RATE,
            'prompt': prompt
        }
        
        return output_path, metadata
        
    except Exception as e:
        return None, str(e)

#############################################
#  API ENDPOINTS
#############################################

@app.route('/health', methods=['GET'])
def health_check():
    """Check if server and model are ready"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'device': device,
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/process', methods=['POST'])
def process():
    """
    Main endpoint to process audio with text prompt
    
    Expected form data:
    - audio: audio file (WAV, MP3, etc.)
    - prompt: text description of effect
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Validate request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if 'prompt' not in request.form:
            return jsonify({'error': 'No prompt provided'}), 400
        
        audio_file = request.files['audio']
        prompt = request.form['prompt']
        
        # Validate file
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            return jsonify({'error': 'Empty prompt'}), 400
        
        if len(prompt) > 200:
            return jsonify({'error': 'Prompt too long (max 200 characters)'}), 400
        
        # Save uploaded file
        upload_filename = f"{uuid.uuid4().hex}_{audio_file.filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
        audio_file.save(upload_path)
        
        print(f"📥 Received: {audio_file.filename}")
        print(f"💬 Prompt: '{prompt}'")
        
        # Process audio
        output_path, metadata = process_audio(upload_path, prompt)
        
        # Clean up upload
        os.remove(upload_path)
        
        if output_path is None:
            return jsonify({'error': f'Processing failed: {metadata}'}), 500
        
        print(f"✅ Processed successfully")
        
        return jsonify({
            'success': True,
            'output_file': os.path.basename(output_path),
            'metadata': metadata,
            'message': 'Audio processed successfully'
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    """Download processed audio file"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"processed_{filename}"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old generated files"""
    try:
        deleted_count = 0
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            os.remove(file_path)
            deleted_count += 1
        
        return jsonify({
            'success': True,
            'deleted_files': deleted_count,
            'message': f'Cleaned up {deleted_count} files'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#############################################
#  RUN SERVER
#############################################

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎵 AUDIO EFFECT BACKEND SERVER")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',  # Accept connections from any IP
        port=5000,
        debug=True,
        threaded=True
    )