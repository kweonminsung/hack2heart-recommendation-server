import random

def create_sample_data():
    """테스트용 샘플 데이터를 생성합니다."""
    # 100명의 사용자 생성
    users = [f'user_{i:03d}' for i in range(100)]
    
    # 기술 스택 및 습관 풀
    languages = ['Python', 'JavaScript', 'TypeScript', 'Java', 'C++', 'Go', 'Rust', 'PHP', 'C#', 'Swift']
    frameworks = ['Django', 'React', 'Vue', 'Angular', 'Spring', 'Flask', 'Express', 'Laravel', 'ASP.NET', 'Rails']
    habits = ['TDD', 'Night-coding', 'Fast-commit', 'Pair-programming', 'Code-review', 'Refactoring', 
             'Documentation', 'Testing', 'Agile', 'Clean-code', 'Git-flow', 'CI-CD']
    
    # 각 사용자에 대해 랜덤한 메타데이터 생성
    user_metadata = {}
    for user in users:
        # 1-2개 언어, 1-2개 프레임워크, 2-4개 습관 선택
        user_languages = random.sample(languages, random.randint(1, 2))
        user_frameworks = random.sample(frameworks, random.randint(1, 2))
        user_habits = random.sample(habits, random.randint(2, 4))
        
        user_metadata[user] = user_languages + user_frameworks + user_habits
    
    # 사용자 간 상호작용 생성 (각 사용자마다 5-15명과 상호작용)
    interactions = []
    for user in users:
        # 본인을 제외한 다른 사용자들 중에서 선택
        other_users = [u for u in users if u != user]
        num_interactions = random.randint(5, 15)
        
        # 중복 없이 선택
        selected_users = random.sample(other_users, min(num_interactions, len(other_users)))
        
        for other_user in selected_users:
            # 평점: 1-5 (1: 매우 불호, 2: 불호, 3: 보통, 4: 호, 5: 매우 호)
            rating = random.randint(1, 5)
            interactions.append((user, other_user, rating))
    
    return users, user_metadata, interactions

def create_small_sample_data():
    """작은 테스트용 샘플 데이터를 생성합니다."""
    users = ['u0', 'u1', 'u2', 'u3']
    
    user_metadata = {
        'u0': ['Python', 'Django', 'TDD', 'Night-coding', 'Fast-commit'],
        'u1': ['JavaScript', 'React', 'Pair-programming', 'TDD', 'Night-coding'],
        'u2': ['Rust', 'Flask', 'Night-coding', 'TDD', 'Pair-programming'],
        'u3': ['Python', 'Flask', 'Fast-commit', 'TDD', 'Night-coding'],
    }
    
    interactions = [
        ('u0', 'u1', 3), ('u0', 'u2', 2), ('u0', 'u3', 1),
        ('u1', 'u0', 3), ('u1', 'u2', 1), ('u1', 'u3', 2),
        ('u2', 'u0', 2), ('u2', 'u1', 3), ('u2', 'u3', 1),
        ('u3', 'u0', 1), ('u3', 'u1', 2), ('u3', 'u2', 3),
    ]
    
    return users, user_metadata, interactions

if __name__ == "__main__":
    # 샘플 데이터 생성 테스트
    users, user_metadata, interactions = create_sample_data()
    print(f"생성된 사용자 수: {len(users)}")
    print(f"생성된 상호작용 수: {len(interactions)}")
    print(f"첫 번째 사용자 메타데이터: {user_metadata[users[0]]}")
    print(f"첫 번째 상호작용: {interactions[0]}")
