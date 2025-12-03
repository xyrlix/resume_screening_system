import os
import random
import string

# 定义行业和岗位类型
industries = ['人工智能', '新能源', '半导体-芯片', '互联网', '电子商务']
job_positions = [
    '算法工程师', '产品经理', '芯片设计工程师', '后端开发工程师', '前端开发工程师',
    '数据工程师', 'DevOps工程师', '机器学习工程师', '电池研发工程师', '跨境电商运营'
]

# 简单的简历模板
resume_template = '''# {name}的简历

## 个人信息
- 姓名：{name}
- 性别：{gender}
- 年龄：{age}
- 电话：138{phone}
- 邮箱：{email}
- 地址：{city}

## 教育背景
- {education}：{university} {major}专业 ({start_year}-{end_year})

## 工作经验
- **{company1}** ({start1}-{end1})
  职位：{position1}
  工作内容：参与{project1}项目，负责{responsibility1}

- **{company2}** ({start2}-{end2})
  职位：{position2}
  工作内容：参与{project2}项目，负责{responsibility2}

## 技能
- 专业技能：{skill1}, {skill2}, {skill3}
- 编程语言：{lang1}, {lang2}, {lang3}
- 工具：{tool1}, {tool2}, {tool3}

## 项目经验
- **{project_name}** ({project_time})
  项目描述：{project_desc}
  负责工作：{project_resp}
  项目成果：{project_result}
'''

# 简单的JD模板
jd_template = '''# {job_title}

## 岗位描述
1. 负责{responsibility1}
2. 参与{responsibility2}
3. 协助{responsibility3}
4. 持续优化{responsibility4}

## 任职要求
1. {requirement1}
2. 熟悉{requirement2}
3. 具备{requirement3}
4. 有{requirement4}优先

## 公司福利
- 五险一金
- 带薪年假
- 定期团建
- 技术培训
'''

# 生成随机姓名
def generate_name():
    last_names = ['王', '李', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴']
    first_names = ['明', '芳', '强', '娟', '军', '磊', '娜', '刚', '霞', '辉',
                   '伟', '秀英', '敏', '静', '丽', '强', '磊', '军', '洋', '勇']
    return random.choice(last_names) + random.choice(first_names)

# 生成随机电话
def generate_phone():
    return ''.join(random.choices(string.digits, k=8))

# 生成随机邮箱
def generate_email(name):
    domains = ['gmail.com', '163.com', '126.com', 'qq.com', 'sina.com']
    return f'{name.lower()}{random.randint(1000, 9999)}@{random.choice(domains)}'

# 生成随机公司名
def generate_company():
    company_prefixes = ['科技', '网络', '数据', '智能', '信息', '软件', '电子', '创新']
    company_suffixes = ['有限公司', '科技有限公司', '网络科技', '信息技术', '软件技术']
    return f'{random.choice(company_prefixes)}{random.choice(company_suffixes)}'

# 生成随机项目名
def generate_project():
    project_prefixes = ['智能', '大数据', '高性能', '分布式', '自动化', '云原生', '移动']
    project_suffixes = ['系统', '平台', '引擎', '工具', '框架', '应用', '解决方案']
    return f'{random.choice(project_prefixes)}{random.choice(project_suffixes)}'

# 生成简历数据
def generate_resume_data():
    name = generate_name()
    return {
        'name': name,
        'gender': random.choice(['男', '女']),
        'age': random.randint(22, 45),
        'phone': generate_phone(),
        'email': generate_email(name),
        'city': random.choice(['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安']),
        'education': random.choice(['本科', '硕士', '博士']),
        'university': random.choice(['清华大学', '北京大学', '浙江大学', '复旦大学', '上海交通大学',
                                   '南京大学', '武汉大学', '华中科技大学', '西安交通大学', '哈尔滨工业大学']),
        'major': random.choice(['计算机科学与技术', '软件工程', '电子信息工程', '自动化', '数据科学',
                              '通信工程', '机械工程', '材料科学与工程', '化学工程', '生物医学工程']),
        'start_year': random.randint(2000, 2015),
        'end_year': random.randint(2004, 2019),
        'company1': generate_company(),
        'company2': generate_company(),
        'start1': f'{random.randint(2015, 2020)}.{random.randint(1, 12)}',
        'end1': f'{random.randint(2018, 2023)}.{random.randint(1, 12)}',
        'start2': f'{random.randint(2020, 2022)}.{random.randint(1, 12)}',
        'end2': '至今',
        'position1': random.choice(job_positions),
        'position2': random.choice(job_positions),
        'project1': generate_project(),
        'project2': generate_project(),
        'responsibility1': random.choice(['系统架构设计', '核心模块开发', '性能优化', '数据处理', 'API设计']),
        'responsibility2': random.choice(['团队管理', '需求分析', '项目规划', '测试优化', '文档编写']),
        'skill1': random.choice(['机器学习', '深度学习', '数据挖掘', '自然语言处理', '计算机视觉']),
        'skill2': random.choice(['微服务架构', '分布式系统', '高并发处理', '容器化技术', '云平台']),
        'skill3': random.choice(['数据库设计', '算法优化', '前端框架', '后端开发', '测试自动化']),
        'lang1': random.choice(['Python', 'Java', 'C++', 'JavaScript', 'Go']),
        'lang2': random.choice(['Python', 'Java', 'C++', 'JavaScript', 'Go', 'PHP', 'Ruby']),
        'lang3': random.choice(['Python', 'Java', 'C++', 'JavaScript', 'Go', 'PHP', 'Ruby', 'Swift']),
        'tool1': random.choice(['Git', 'Docker', 'Kubernetes', 'Jenkins', 'Linux']),
        'tool2': random.choice(['MySQL', 'MongoDB', 'Redis', 'Elasticsearch', 'PostgreSQL']),
        'tool3': random.choice(['TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy']),
        'project_name': generate_project(),
        'project_time': f'{random.randint(2021, 2023)}年',
        'project_desc': f'{random.choice(["基于AI的", "大数据驱动的", "高性能的", "分布式的"])}系统开发',
        'project_resp': random.choice(['核心算法开发', '系统架构设计', '前端界面开发', '后端服务实现']),
        'project_result': f'提高了{random.randint(10, 50)}%的效率，节省了{random.randint(5, 20)}%的成本'
    }

# 生成JD数据
def generate_jd_data():
    industry = random.choice(industries)
    position = random.choice(job_positions)
    job_title = f'{position}'
    
    return {
        'job_title': job_title,
        'responsibility1': random.choice([
            '公司核心产品的研发工作', '项目的需求分析和技术方案设计',
            '系统架构的设计和优化', '核心模块的开发和维护'
        ]),
        'responsibility2': random.choice([
            '技术团队的管理和指导', '与其他部门的沟通和协作',
            '技术文档的编写和维护', '技术难题的解决和攻关'
        ]),
        'responsibility3': random.choice([
            '新技术的调研和引入', '产品功能的优化和改进',
            '项目进度的跟踪和控制', '质量保证和测试工作'
        ]),
        'responsibility4': random.choice([
            '系统性能和稳定性', '代码质量和开发效率',
            '用户体验和产品功能', '团队协作和沟通效率'
        ]),
        'requirement1': random.choice([
            '本科及以上学历，计算机相关专业', '硕士及以上学历，相关专业',
            '3年以上相关工作经验', '5年以上相关工作经验'
        ]),
        'requirement2': random.choice([
            'Python/Java/C++等编程语言', '机器学习/深度学习算法',
            '分布式系统设计', '微服务架构'
        ]),
        'requirement3': random.choice([
            '良好的问题分析和解决能力', '优秀的团队协作和沟通能力',
            '扎实的计算机基础知识', '较强的学习能力和创新精神'
        ]),
        'requirement4': random.choice([
            '大型项目经验', '开源项目贡献', '相关领域论文发表', '行业认证证书'
        ])
    }

# 生成简历文件
def generate_resumes(target_count=500):
    resume_dir = 'f:/output/resume_screening_system/data/raw_resumes/'
    formats = ['md', 'txt', 'pdf', 'png']  # 选择需要生成的格式
    
    # 统计现有简历数量
    current_count = 0
    for fmt in formats:
        current_count += len(os.listdir(os.path.join(resume_dir, fmt)))
    
    # 需要生成的简历数量
    need_generate = max(0, target_count - current_count)
    print(f'需要生成{need_generate}份简历')
    
    if need_generate <= 0:
        print('现有简历数量已满足要求')
        return
    
    # 平均分配到各个格式
    per_format = need_generate // len(formats)
    remainder = need_generate % len(formats)
    
    for i, fmt in enumerate(formats):
        count = per_format + (1 if i < remainder else 0)
        if count <= 0:
            continue
            
        format_dir = os.path.join(resume_dir, fmt)
        # 获取现有文件数量
        existing = len(os.listdir(format_dir))
        
        for j in range(count):
            # 生成简历数据
            resume_data = generate_resume_data()
            filename = f'resume_{existing + j}.{fmt}'
            filepath = os.path.join(format_dir, filename)
            
            # 生成文件内容
            content = resume_template.format(**resume_data)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f'生成简历: {filepath}')

# 生成JD文件
def generate_jds(target_count=100):
    jd_dir = 'f:/output/resume_screening_system/data/raw_jobs/'
    
    # 统计现有JD数量
    current_count = len([f for f in os.listdir(jd_dir) if f.endswith('.txt')])
    
    # 需要生成的JD数量
    need_generate = max(0, target_count - current_count)
    print(f'需要生成{need_generate}份JD')
    
    if need_generate <= 0:
        print('现有JD数量已满足要求')
        return
    
    for i in range(need_generate):
        # 生成JD数据
        jd_data = generate_jd_data()
        industry = random.choice(industries)
        position = jd_data['job_title']
        filename = f'{industry}_{position}_job_{current_count + i}.txt'
        filepath = os.path.join(jd_dir, filename)
        
        # 生成文件内容
        content = jd_template.format(**jd_data)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f'生成JD: {filepath}')

if __name__ == '__main__':
    print('开始生成简历数据...')
    generate_resumes()
    print('\n开始生成JD数据...')
    generate_jds()
    print('\n数据生成完成！')