import gdown

# download links for GPT substitutions + augmented data + bleu alignment
# this is our best performing dataset for which we also report the results in the paper
datasets = [
    'https://drive.google.com/uc?id=1u6EHk9fVclyAqBtC2Qyxdu4g8_EpJeJe', # Baseline train
    'https://drive.google.com/uc?id=1aR4ybwTi6DzMsHhh4rQZVnTeaURfOcdN', # Baseline test
    'https://drive.google.com/uc?id=10OVYAfXhXa-aFCnTbHczkWxNohmuCQM9', # Baseline dev
    'https://drive.google.com/uc?id=13SjJ4QyKABtupA0rmx8vuxQwyJPclSbf', # HandsAndWholeData train
    'https://drive.google.com/uc?id=1INpTE0vUKj2DxIJEZFUh4UCuiDp2exYC', # HandsAndWholeData test
    'https://drive.google.com/uc?id=1iP7ba0KQQ31--PNu3ZVZSjjPke1wZbao', # HandsAndWholeData dev
    'https://drive.google.com/uc?id=11cl9kjcW3IPw84gLk7-SwsBQYMEcwiw8', # MouthAndWholeData train
    'https://drive.google.com/uc?id=1zDkeL1C5t9uQWr0qYF8_QsNLPCTZUwlF', # MouthAndWholeData test
    'https://drive.google.com/uc?id=19T_qqVD2ft1sZzsKhfuAq3dHBhNS6AKo', # MouthAndWholeData dev
    'https://drive.google.com/uc?id=1qLSluNUMywQNRNqRueImH59spbYByw2n', # HandsMouthAndWholeData train
    'https://drive.google.com/uc?id=1XzqZ7UX7UNyFNgbD6kcooLbVv0OCDJhQ', # HandsMouthAndWholeData test
    'https://drive.google.com/uc?id=11nann53Ntd3ly9LzlyomGCUGJxQ16KEI', # HandsMouthAndWholeData dev
    'https://drive.google.com/uc?id=1Jx5W6f3xD0c5vHpTKwD-96bRSNbtuEgZ', # RightHandAndWholeData train
    'https://drive.google.com/uc?id=1gnxnpi6a5ntAviabFE32vCUIqY3VXAcL', # RightHandAndWholeData test
    'https://drive.google.com/uc?id=1GamKKx2s4aJs1Xa-O9SF1MBykHMRuww_', # RightHandAndWholeData dev
    'https://drive.google.com/uc?id=1nUe49pdCuV7MJZBKn4Ovqo0hkLAsognB', # LeftHandAndWholeData train
    'https://drive.google.com/uc?id=1PVfnWVx1mmZNqphOjM3wvx3NEHA_1W5m', # LeftHandAndWholeData test
    'https://drive.google.com/uc?id=10LYy9Gx3Ictz59gPRHAK_NOcnWalNJRO', # LeftHandAndWholeData dev
]


if __name__ == "__main__":
    for dataset_url in datasets:
        gdown.download(dataset_url)
