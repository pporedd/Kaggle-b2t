import pandas as pd
import numpy as np

# Re-defining the Master Data to ensure we have the full dataset available to print
master_data = [
    # --- VOWELS (Monophthongs) ---
    {'Arpabet': 'AA', 'IPA': 'ɑ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Relaxed', 'Jaw': 'Open', 'Tongue': 'Low Back'},
    {'Arpabet': 'AE', 'IPA': 'æ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Spread', 'Jaw': 'Open-Mid', 'Tongue': 'Low Front'},
    {'Arpabet': 'AH', 'IPA': 'ʌ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Open-Mid', 'Tongue': 'Mid Back'},
    {'Arpabet': 'AO', 'IPA': 'ɔ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Rounded', 'Jaw': 'Open-Mid', 'Tongue': 'Low Back'}, 
    {'Arpabet': 'EH', 'IPA': 'ɛ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Open-Mid', 'Tongue': 'Mid Front'},
    {'Arpabet': 'ER', 'IPA': 'ɝ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Rounded', 'Jaw': 'Mid', 'Tongue': 'Mid Central'}, 
    {'Arpabet': 'IH', 'IPA': 'ɪ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.6,
     'Lips': 'Spread', 'Jaw': 'Close-Mid', 'Tongue': 'High Front Lax'},
    {'Arpabet': 'IY', 'IPA': 'i', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Spread', 'Jaw': 'Close', 'Tongue': 'High Front'},
    {'Arpabet': 'UH', 'IPA': 'ʊ', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.6,
     'Lips': 'Rounded', 'Jaw': 'Close-Mid', 'Tongue': 'High Back Lax'},
    {'Arpabet': 'UW', 'IPA': 'u', 'Type': 'Monophthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Protruded', 'Jaw': 'Close', 'Tongue': 'High Back'},

    # --- DIPHTHONGS (Dynamic) ---
    {'Arpabet': 'AW', 'IPA': 'aʊ', 'Type': 'Diphthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Dynamic', 'Duration': 1.2,
     'Start_Lips': 'Relaxed', 'Start_Jaw': 'Open', 'Start_Tongue': 'Low Central',
     'End_Lips': 'Rounded', 'End_Jaw': 'Close-Mid', 'End_Tongue': 'High Back Lax'},
    {'Arpabet': 'AY', 'IPA': 'aɪ', 'Type': 'Diphthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Dynamic', 'Duration': 1.2,
     'Start_Lips': 'Relaxed', 'Start_Jaw': 'Open', 'Start_Tongue': 'Low Central',
     'End_Lips': 'Spread', 'End_Jaw': 'Close-Mid', 'End_Tongue': 'High Front Lax'},
    {'Arpabet': 'OY', 'IPA': 'ɔɪ', 'Type': 'Diphthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Dynamic', 'Duration': 1.2,
     'Start_Lips': 'Rounded', 'Start_Jaw': 'Open-Mid', 'Start_Tongue': 'Low Back',
     'End_Lips': 'Spread', 'End_Jaw': 'Close-Mid', 'End_Tongue': 'High Front Lax'},
    {'Arpabet': 'EY', 'IPA': 'eɪ', 'Type': 'Diphthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Dynamic', 'Duration': 1.1,
     'Start_Lips': 'Spread', 'Start_Jaw': 'Mid', 'Start_Tongue': 'Mid Front',
     'End_Lips': 'Spread', 'End_Jaw': 'Close-Mid', 'End_Tongue': 'High Front Lax'},
    {'Arpabet': 'OW', 'IPA': 'oʊ', 'Type': 'Diphthong', 'Voiced': True, 'Nasal': False,
     'Transition': 'Dynamic', 'Duration': 1.1,
     'Start_Lips': 'Rounded', 'Start_Jaw': 'Mid', 'Start_Tongue': 'Mid Back',
     'End_Lips': 'Protruded', 'End_Jaw': 'Close-Mid', 'End_Tongue': 'High Back Lax'},

    # --- STOPS ---
    {'Arpabet': 'B', 'IPA': 'b', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Bilabial Close', 'Jaw': 'Close-Mid', 'Tongue': 'Neutral'},
    {'Arpabet': 'D', 'IPA': 'd', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Alveolar Stop'},
    {'Arpabet': 'G', 'IPA': 'g', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Velar Stop'},
    {'Arpabet': 'P', 'IPA': 'p', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Bilabial Close', 'Jaw': 'Close-Mid', 'Tongue': 'Neutral'},
    {'Arpabet': 'T', 'IPA': 't', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Alveolar Stop'},
    {'Arpabet': 'K', 'IPA': 'k', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Velar Stop'},

    # --- FRICATIVES ---
    {'Arpabet': 'F', 'IPA': 'f', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Labiodental', 'Jaw': 'Close-Mid', 'Tongue': 'Neutral'},
    {'Arpabet': 'V', 'IPA': 'v', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Labiodental', 'Jaw': 'Close-Mid', 'Tongue': 'Neutral'},
    {'Arpabet': 'TH', 'IPA': 'θ', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Dental Fricative'},
    {'Arpabet': 'DH', 'IPA': 'ð', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Dental Fricative'},
    {'Arpabet': 'S', 'IPA': 's', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Spread', 'Jaw': 'Close', 'Tongue': 'Alveolar Fricative'},
    {'Arpabet': 'Z', 'IPA': 'z', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Spread', 'Jaw': 'Close', 'Tongue': 'Alveolar Fricative'},
    {'Arpabet': 'SH', 'IPA': 'ʃ', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Protruded', 'Jaw': 'Close', 'Tongue': 'Palatal Fricative'},
    {'Arpabet': 'ZH', 'IPA': 'ʒ', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Protruded', 'Jaw': 'Close', 'Tongue': 'Palatal Fricative'},
    {'Arpabet': 'HH', 'IPA': 'h', 'Type': 'Consonant', 'Voiced': False, 'Nasal': False,
     'Transition': 'Static', 'Duration': 0.8,
     'Lips': 'Relaxed', 'Jaw': 'Open-Mid', 'Tongue': 'Neutral'},

    # --- NASALS ---
    {'Arpabet': 'M', 'IPA': 'm', 'Type': 'Consonant', 'Voiced': True, 'Nasal': True,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Bilabial Close', 'Jaw': 'Close-Mid', 'Tongue': 'Neutral'},
    {'Arpabet': 'N', 'IPA': 'n', 'Type': 'Consonant', 'Voiced': True, 'Nasal': True,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Alveolar Stop'},
    {'Arpabet': 'NG', 'IPA': 'ŋ', 'Type': 'Consonant', 'Voiced': True, 'Nasal': True,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Velar Stop'},

    # --- LIQUIDS ---
    {'Arpabet': 'L', 'IPA': 'l', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Relaxed', 'Jaw': 'Close-Mid', 'Tongue': 'Alveolar Lateral'},
    {'Arpabet': 'R', 'IPA': 'ɹ', 'Type': 'Consonant', 'Voiced': True, 'Nasal': False,
     'Transition': 'Static', 'Duration': 1.0,
     'Lips': 'Protruded', 'Jaw': 'Close-Mid', 'Tongue': 'Retroflex'},
]

def get_muscle_vector(target_lips, target_jaw, target_tongue, is_voiced, is_nasal):
    muscles = {
        'Lip_OO': 0.0, 'Lip_Ris': 0.0,
        'Jaw_Mass': 0.0, 'Jaw_Dig': 0.0,
        'Tip_SupLong': 0.0,
        'Body_GGp': 0.0, 'Body_GGa': 0.0, 'Body_SG': 0.0, 'Body_HG': 0.0,
        'Velum_Lev': 0.0, 'Velum_Pal': 0.0,
        'Larynx_LCA': 0.0, 'Larynx_PCA': 0.0,
        # --- NEW DIMENSIONS (Added at the end) ---
        'Lip_LAO': 0.0,      # Levator Anguli Oris
        'Larynx_CT': 0.0,    # Cricothyroid (Pitch)
        'Larynx_Strap': 0.0  # Sternohyoid (Strap)
    }

    # --- LARYNX ---
    if is_voiced: 
        muscles['Larynx_LCA'] = 0.8
        muscles['Larynx_CT'] = 0.3 # Base voicing tension
    else: 
        muscles['Larynx_PCA'] = 0.8

    # --- VELUM ---
    if is_nasal:
        muscles['Velum_Pal'] = 0.9 
        muscles['Velum_Lev'] = 0.0
    else:
        muscles['Velum_Lev'] = 0.9 
    
    # --- LIPS ---
    if target_lips == 'Rounded': muscles['Lip_OO'] = 0.8
    elif target_lips == 'Protruded': muscles['Lip_OO'] = 0.9
    elif target_lips == 'Bilabial Close': 
        muscles['Lip_OO'] = 1.0
        muscles['Lip_LAO'] = 0.2
    elif target_lips == 'Spread': 
        muscles['Lip_Ris'] = 0.8
        muscles['Lip_LAO'] = 0.4
    elif target_lips == 'Labiodental': 
        muscles['Lip_OO'] = 0.4; muscles['Lip_Ris'] = 0.2
        muscles['Lip_LAO'] = 0.6

    # --- JAW ---
    if target_jaw == 'Close': muscles['Jaw_Mass'] = 0.8
    elif target_jaw == 'Close-Mid': muscles['Jaw_Mass'] = 0.5
    elif target_jaw == 'Mid': muscles['Jaw_Mass'] = 0.2
    elif target_jaw == 'Open-Mid': muscles['Jaw_Dig'] = 0.4
    elif target_jaw == 'Open': muscles['Jaw_Dig'] = 0.9

    # --- TONGUE & TUNING ---
    tgt_str = str(target_tongue)
    
    if target_tongue == 'High Front': 
        muscles['Body_GGp'] = 0.9; muscles['Jaw_Mass'] = max(muscles['Jaw_Mass'], 0.6)
        if is_voiced: muscles['Larynx_CT'] = 0.7
    elif target_tongue == 'High Front Lax': 
        muscles['Body_GGp'] = 0.6
        if is_voiced: muscles['Larynx_CT'] = 0.6
    elif target_tongue == 'Mid Front': 
        muscles['Body_GGp'] = 0.5; muscles['Body_GGa'] = 0.2
        if is_voiced: muscles['Larynx_CT'] = 0.5
    elif target_tongue == 'Low Front': 
        muscles['Body_GGp'] = 0.4; muscles['Body_GGa'] = 0.6; muscles['Body_HG'] = 0.4
        muscles['Larynx_Strap'] = 0.5
        if is_voiced: muscles['Larynx_CT'] = 0.2
    elif target_tongue == 'Mid Back': 
        muscles['Body_SG'] = 0.4; muscles['Body_HG'] = 0.4
        if is_voiced: muscles['Larynx_CT'] = 0.4
    elif target_tongue == 'Low Back': 
        muscles['Body_HG'] = 0.9; muscles['Body_SG'] = 0.3
        muscles['Larynx_Strap'] = 0.7
        if is_voiced: muscles['Larynx_CT'] = 0.1
    elif target_tongue == 'High Back': 
        muscles['Body_SG'] = 0.9; muscles['Body_GGp'] = 0.2
        if is_voiced: muscles['Larynx_CT'] = 0.7
    elif target_tongue == 'High Back Lax': 
        muscles['Body_SG'] = 0.6
        if is_voiced: muscles['Larynx_CT'] = 0.6
    elif target_tongue == 'Alveolar Stop' or target_tongue == 'Alveolar Lateral': 
        muscles['Tip_SupLong'] = 0.9
    elif target_tongue == 'Alveolar Fricative': 
        muscles['Tip_SupLong'] = 0.8; muscles['Body_GGp'] = 0.3
        muscles['Lip_LAO'] = 0.3
    elif target_tongue == 'Palatal Fricative': 
        muscles['Tip_SupLong'] = 0.6; muscles['Body_GGp'] = 0.7
        muscles['Lip_LAO'] = 0.4
    elif target_tongue == 'Velar Stop': 
        muscles['Body_SG'] = 0.9; muscles['Velum_Pal'] = max(muscles['Velum_Pal'], 0.4)
    elif target_tongue == 'Dental Fricative': 
        muscles['Tip_SupLong'] = 0.4
        muscles['Lip_LAO'] = 0.3
    elif target_tongue == 'Retroflex': 
        muscles['Tip_SupLong'] = 1.0; muscles['Body_SG'] = 0.5
    
    return muscles

rows = []
for item in master_data:
    row = {
        'Arpabet': item['Arpabet'],
        'IPA': item['IPA'],
        'Type': item['Type'],
        'Transition_Type': item['Transition'],
        'Duration_Factor': item['Duration']
    }
    
    if item['Transition'] == 'Static':
        s_lips = e_lips = item.get('Lips')
        s_jaw = e_jaw = item.get('Jaw')
        s_tongue = e_tongue = item.get('Tongue')
    else:
        s_lips = item.get('Start_Lips'); e_lips = item.get('End_Lips')
        s_jaw = item.get('Start_Jaw'); e_jaw = item.get('End_Jaw')
        s_tongue = item.get('Start_Tongue'); e_tongue = item.get('End_Tongue')

    start_vec = get_muscle_vector(s_lips, s_jaw, s_tongue, item['Voiced'], item['Nasal'])
    end_vec = get_muscle_vector(e_lips, e_jaw, e_tongue, item['Voiced'], item['Nasal'])

    for k, v in start_vec.items(): row[f'Start_{k}'] = v
    for k, v in end_vec.items(): row[f'End_{k}'] = v
        
    rows.append(row)

df_final = pd.DataFrame(rows)
df_final.to_csv('bodyVector.csv', index=False)