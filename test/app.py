"""
Flask API for DAC-VAE Audio Effect Generator
Receives audio file + prompt, returns processed audio
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import tempfile
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
from einops import rearrange
from werkzeug.utils import secure_filename
import traceback

# DAC import
try:
    import dac
    print("✓ DAC library imported successfully")
except ImportError:
    print("❌ DAC not installed. Run: pip install descript-audio-codec")
    exit(1)

#############################################
#     MODEL ARCHITECTURE (SAME AS TRAINING)
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
    def __init__(self, in_c, out_c, text_dim=768, use_attn=False):
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

class LatentUNet(nn.Module):
    def __init__(self, latent_channels, channels, text_dim=768):
        super().__init__()
        self.input_conv = nn.Conv1d(latent_channels, channels[0], 7, padding=3)
        
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            use_attn = i >= 2
            self.down_blocks.append(DownBlock(channels[i], channels[i+1], text_dim, use_attn))
        
        self.mid_block1 = ResidualBlock(channels[-1])
        self.mid_attn = CrossAttention(channels[-1], text_dim)
        self.mid_block2 = ResidualBlock(channels[-1])
        
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            use_attn = i >= 2
            self.up_blocks.append(
                UpBlock(channels[i], channels[i-1], channels[i], text_dim, use_attn)
            )
        
        self.output_conv = nn.Conv1d(channels[0], latent_channels, 7, padding=3)
        
    def forward(self, z, text_emb):
        original_length = z.size(-1)
        x = self.input_conv(z)
        
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

class AudioEffectModel(nn.Module):
    def __init__(self, dac_model, latent_channels, unet_channels, text_dim):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.dac = dac_model
        self.unet = LatentUNet(latent_channels, unet_channels, text_dim)
        
    @torch.no_grad()
    def generate(self, wav_in, prompt, sample_rate, tokenizer):
        self.eval()
        
        if wav_in.dim() == 2:
            wav_in = wav_in.unsqueeze(1)
        
        tokens = tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(wav_in.device)
        
        text_output = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask
        )
        text_emb = text_output.last_hidden_state
        
        z_in = self.dac.encoder(wav_in)
        z_out = self.unet(z_in, text_emb)
        wav_out = self.dac.decoder(z_out)
        
        return wav_out

#############################################
#     FLASK APP
#############################################

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global variables for model
model = None
tokenizer = None
device = None
sample_rate = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(model_path, dac_model_path=None, use_device='cuda'):
    """Load model at startup"""
    global model, tokenizer, device, sample_rate
    
    device = use_device if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print("LOADING MODEL FOR FLASK API")
    print(f"{'='*60}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    
    config = ckpt['config']
    sample_rate = config['sample_rate']
    latent_channels = config['latent_channels']
    unet_channels = config['unet_channels']
    text_dim = config['text_dim']
    
    print(f"✓ Sample rate: {sample_rate} Hz")
    print(f"✓ Latent channels: {latent_channels}")
    print(f"✓ UNet channels: {unet_channels}")
    
    # Load DAC model
    if dac_model_path is None:
        dac_model_path = "../models/weights_44khz_8kbps_0.0.1.pth"
    
    print(f"Loading DAC model from: {dac_model_path}")
    
    if not os.path.exists(dac_model_path):
        raise FileNotFoundError(f"DAC model not found at: {dac_model_path}")
    
    dac_model = dac.DAC.load(dac_model_path)
    dac_model = dac_model.to(device)
    dac_model.eval()
    print("✓ DAC model loaded")
    
    # Create model
    model = AudioEffectModel(
        dac_model=dac_model,
        latent_channels=latent_channels,
        unet_channels=unet_channels,
        text_dim=text_dim
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("✓ Model weights loaded")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("✓ Tokenizer loaded")
    print(f"✓ Device: {device}")
    print(f"{'='*60}\n")

def process_audio_file(input_path, prompt):
    """Process audio file and return output path"""
    global model, tokenizer, device, sample_rate
    
    # Load audio
    wav, sr = sf.read(input_path)
    wav = torch.from_numpy(wav).float()
    
    # Ensure correct shape
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.size(0) > wav.size(1):
        wav = wav.t()
    
    # Resample if needed
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    
    # Convert to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Add batch dimension and move to device
    wav = wav.unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        wav_out = model.generate(wav, prompt, sample_rate, tokenizer)
    
    # Move to CPU
    wav_out = wav_out.squeeze(0).cpu()
    
    # Match original length
    target_length = wav.squeeze(0).size(-1)
    current_length = wav_out.size(-1)
    
    if current_length != target_length:
        if current_length > target_length:
            wav_out = wav_out[..., :target_length]
        else:
            wav_out = F.pad(wav_out, (0, target_length - current_length))
    
    # Save output
    output_filename = f"output_{os.path.basename(input_path)}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    wav_out_np = wav_out.squeeze(0).numpy()
    sf.write(output_path, wav_out_np, sample_rate)
    
    return output_path

#############################################
#     API ROUTES
#############################################

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'sample_rate': sample_rate
    })

@app.route('/process', methods=['POST'])
def process_audio():
    """
    Main endpoint to process audio
    
    Expected form data:
    - audio: audio file
    - prompt: text description of effect
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': f'File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}'}), 400
        
        # Get prompt
        prompt = request.form.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(input_path)
        
        print(f"Processing: {filename}")
        print(f"Prompt: {prompt}")
        
        # Process audio
        output_path = process_audio_file(input_path, prompt)
        
        # Clean up input file
        os.remove(input_path)
        
        # Send output file
        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'processed_{filename}'
        )
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/process-url', methods=['POST'])
def process_audio_url():
    """
    Alternative endpoint that returns a URL to download the processed audio
    Useful for frontend that wants to handle download separately
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': f'File type not allowed'}), 400
        
        prompt = request.form.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(input_path)
        
        # Process audio
        output_path = process_audio_file(input_path, prompt)
        output_filename = os.path.basename(output_path)
        
        # Clean up input file
        os.remove(input_path)
        
        return jsonify({
            'success': True,
            'message': 'Audio processed successfully',
            'download_url': f'/download/{output_filename}'
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed audio file"""
    try:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(output_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#############################################
#     MAIN
#############################################

if __name__ == '__main__':
    import sys
    
    # Get model path from command line
    if len(sys.argv) < 2:
        print("Usage: python flask_app.py <model_path> [dac_model_path] [port]")
        print("Example: python flask_app.py model_best.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    dac_model_path = sys.argv[2] if len(sys.argv) > 2 else None
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    # Load model
    try:
        load_model(model_path, dac_model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Run Flask app
    print(f"\n🚀 Starting Flask server on port {port}")
    print(f"API Endpoints:")
    print(f"  - Health check: http://localhost:{port}/health")
    print(f"  - Process audio: http://localhost:{port}/process (POST)")
    print(f"  - Process with URL: http://localhost:{port}/process-url (POST)")
    print(f"\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)