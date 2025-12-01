import pandas as pd

sound_effect_prompts = {
    "rain": [
        "Add gentle rain falling in the background",
        "Introduce a calm rainfall ambience",
        "Overlay soft raindrops hitting the ground",
        "Add light rain tapping on a window",
        "Include distant rain sounds for a peaceful tone",
        "Add steady rainfall like during a quiet storm",
        "Mix in background rain ambience",
        "Blend subtle rain falling throughout the track",
        "Add continuous rain for a relaxed atmosphere",
        "Introduce a light drizzle of rain across the scene"
    ],
    "thunderstorm": [
        "Add distant thunder rolling in the background",
        "Introduce a calm thunderstorm ambience",
        "Overlay low rumbling thunder behind the audio",
        "Add occasional thunderclaps softly",
        "Include background stormy weather sounds",
        "Mix in deep thunder echoing in the distance",
        "Add slow thunder rumbles across the sky",
        "Introduce a light storm atmosphere",
        "Blend subtle thunder and rain for tension",
        "Add faint thunderstorm ambience for realism"
    ],
    "drizzle": [
        "Add light drizzle sounds in the background",
        "Introduce gentle raindrops softly falling",
        "Overlay a fine mist of rain",
        "Add a calm drizzle ambience",
        "Include quiet rain sounds like a light shower",
        "Mix in a delicate drizzle effect",
        "Add faint rain patter throughout",
        "Introduce subtle raindrops on leaves",
        "Blend continuous drizzle ambience",
        "Add soft background rain for a tranquil mood"
    ],
    "hail": [
        "Add the sound of hail hitting the ground",
        "Introduce light hail tapping on a surface",
        "Overlay soft hailstone impacts in the background",
        "Add gentle hail sounds under the audio",
        "Include distant hailstorm ambience",
        "Mix in quick hail drops falling",
        "Add a realistic hail effect during rain",
        "Introduce short bursts of hail impacts",
        "Blend subtle hail sounds for atmosphere",
        "Add a background hail layer for realism"
    ],
    "snow_falling": [
        "Add the soft sound of snow falling quietly",
        "Introduce gentle winter ambience",
        "Overlay faint snow sounds for a calm mood",
        "Add background snow atmosphere",
        "Include light snowflakes landing softly",
        "Mix in a peaceful snowy ambience",
        "Add subtle winter wind and snow sounds",
        "Introduce slow snow falling ambience",
        "Blend delicate snow sounds under the audio",
        "Add a serene snowfall background layer"
    ],
    "wind": [
        "Add soft wind blowing in the background",
        "Introduce a gentle breeze ambience",
        "Overlay the sound of air moving calmly",
        "Add faint outdoor wind noise",
        "Include light gusts passing through trees",
        "Mix in a continuous soft wind flow",
        "Add distant wind ambience for realism",
        "Introduce smooth air movement sounds",
        "Blend steady wind breeze behind the track",
        "Add calm wind ambience for atmosphere"
    ],
    "leaves_rustling": [
        "Add soft rustling leaves in the background",
        "Introduce the sound of wind passing through trees",
        "Overlay gentle leaf movement",
        "Add a calm forest breeze rustling leaves",
        "Include light leaf noise for a natural feel",
        "Mix in subtle rustling ambience",
        "Add faint leaves brushing together",
        "Introduce forest leaves swaying softly",
        "Blend background rustling for realism",
        "Add a quiet foliage movement layer"
    ],
    "forest_ambience": [
        "Add peaceful forest ambience to the audio",
        "Introduce distant birds and wind in the forest",
        "Overlay natural woodland background sounds",
        "Add subtle forest noise for depth",
        "Include calm outdoor forest atmosphere",
        "Mix in gentle forest breeze and birds",
        "Add background forest ambience layer",
        "Introduce tranquil woodland environment",
        "Blend forest soundscape under the track",
        "Add natural forest sounds for a serene tone"
    ],
    "river_flowing": [
        "Add gentle river flowing in the background",
        "Introduce calm water stream ambience",
        "Overlay soft river sounds for atmosphere",
        "Add smooth water flow behind the audio",
        "Include light bubbling stream noises",
        "Mix in quiet river movement sounds",
        "Add distant flowing water ambience",
        "Introduce natural stream background",
        "Blend continuous river flow under the track",
        "Add a peaceful river sound effect layer"
    ],
    "waterfall": [
        "Add distant waterfall sounds in the background",
        "Introduce a calm waterfall ambience",
        "Overlay rushing water for a natural tone",
        "Add smooth waterfall noise under the audio",
        "Include background waterfall environment",
        "Mix in soft cascading water sounds",
        "Add a light waterfall ambience layer",
        "Introduce continuous water falling sounds",
        "Blend natural waterfall noise softly",
        "Add peaceful waterfall ambience throughout"
    ]
}

rows = []
for sound_effect, prompts in sound_effect_prompts.items():
    for prompt in prompts:
        rows.append({"sound_effect": sound_effect, "prompt": prompt})

df = pd.DataFrame(rows)
df.to_excel("prompts.xlsx", index=False)

print("CSV file 'prompts.csv' created successfully.")