import os
import glob

# 统计现有简历数量
print('=== 现有简历数量统计 ===')
resume_paths = ['docx', 'md', 'pdf', 'png', 'txt', 'xlsx']
total_resumes = 0

for path in resume_paths:
    files = glob.glob(f'f:/output/resume_screening_system/data/raw_resumes/{path}/*')
    count = len(files)
    total_resumes += count
    print(f'{path}: {count}份')

print(f'总计: {total_resumes}份\n')

# 统计现有JD数量
print('=== 现有JD数量统计 ===')
jd_files = glob.glob('f:/output/resume_screening_system/data/raw_jobs/*.txt')
print(f'JD总数: {len(jd_files)}份')