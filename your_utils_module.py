def infer_mbti_from_mood(mood_distribution):
    mbti_scores = {
        'E': 0, 'I': 0,
        'S': 0, 'N': 0,
        'T': 0, 'F': 0,
        'J': 0, 'P': 0
    }

    trait_map = {
        'joyful':     ['E', 'F', 'P'],
        'energetic':  ['E', 'S', 'J'],
        'calm':       ['I', 'F', 'P'],
        'sad':        ['I', 'F'],
        'melancholic':['I', 'N', 'T'],
        'intense':    ['E', 'T', 'J'],
        'relaxed':    ['P', 'F', 'N'],
        'hype':       ['E', 'S', 'J']
    }

    for mood, weight in mood_distribution.items():
        traits = trait_map.get(mood, [])
        for trait in traits:
            mbti_scores[trait] += weight

    mbti = ''
    mbti += 'E' if mbti_scores['E'] >= mbti_scores['I'] else 'I'
    mbti += 'S' if mbti_scores['S'] >= mbti_scores['N'] else 'N'
    mbti += 'T' if mbti_scores['T'] >= mbti_scores['F'] else 'F'
    mbti += 'J' if mbti_scores['J'] >= mbti_scores['P'] else 'P'

    return mbti
