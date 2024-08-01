import gdown

# TODO: add only datasets for which we report results in the final paper
datasets = [
    'https://drive.google.com/uc?id=1wRwKE7A4eyXWvcI6Qb5xrfJmltTMjSBm', # Baseline train
    'https://drive.google.com/uc?id=1l9J-ojZ1LaM70pOYS7YMqqEA7252AxxN', # Baseline test
    'https://drive.google.com/uc?id=1DPAe3807kDr1kAui5S8ZK8x2yx85lIL4', # Baseline dev
    'https://drive.google.com/uc?id=1-H7WGFtiPwb0U7lPa6oaFcnOLjMeqEOH', # HandsAndWholeDataGPT train
    'https://drive.google.com/uc?id=1-PiHfNcaWho6wVY3x6kFEEviurvzoUVd', # HandsAndWholeDataGPT test
    'https://drive.google.com/uc?id=1-tze5w6w-UTLU2QKazmTRJW9LCS95uj9', # HandsAndWholeDataGPT dev
]


if __name__ == "__main__":
    for dataset_url in datasets:
        gdown.download(dataset_url)
