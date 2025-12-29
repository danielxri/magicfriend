import os

meta_path = 'ai_service/MuseTalk/dataset/HDTF/meta'
train_txt_path = 'train_full.txt' # Write to CWD

files = [f for f in os.listdir(meta_path) if f.endswith('.json')]
print(f'Found {len(files)} files')

with open(train_txt_path, 'w') as f:
    f.write('HEADER\n')
    f.write('\n'.join(files))

print(f'Successfully wrote to {train_txt_path}')
