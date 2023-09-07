def select_type():
    type_options = ['UNICEF', 'SPA', 'JPNY', 'EUHR', 'CHNFA', 'CHNF', 'ENG']
    print("유형을 선택하세요:")
    for idx, option in enumerate(type_options):
        print(f"{idx + 1}. {option}")
    type_choice_idx = int(input("유형 번호를 입력하세요: ")) - 1
    return type_options[type_choice_idx]
