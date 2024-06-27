import albumentations as albu

A_transform = albu.Compose([
        # 隨機水平翻轉圖像
        albu.HorizontalFlip(p=0.5), 
        
        # 可以進行平移、縮放和旋轉操作。 
        # scale_limit=0: 不進行縮放操作。
        # rotate_limit=0: 不進行旋轉操作。
        # shift_limit=0.1: 圖像會隨機平移，平移幅度的上限為圖像尺寸的 10%。
        # p=1: 表示這個變換會 100% 被應用。
        # border_mode=0: 使用常數填充邊界區域，填充值為 0 (黑色)。
        albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    
        # 在圖像中添加高斯噪聲。p=0.1 表示有 10% 的機率添加高斯噪聲。
        albu.GaussNoise(p=0.1),
    
        # 將圖像進行透視變換，改變圖像的視角。p=0.5 表示有 50% 的機率應用透視變換。
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                # CLAHE 強度
                albu.CLAHE(p=1),
                
                # 隨機調整亮度和對比度。p=1 表示總是應用此變換。
                albu.RandomBrightnessContrast(p=1),
                
                # 隨機調整圖像的伽馬值。p=1 表示總是應用此變換。
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # 對圖像進行銳化處理。p=1 表示總是應用此變換。
                albu.Sharpen(p=1),
                
                # 隨機對圖像進行模糊處理，模糊範圍最大為 3 像素。p=1 表示總是應用此變換。
                albu.Blur(blur_limit=3, p=1),
                
                # 模擬運動模糊，模糊範圍最大為 3 像素。p=1 表示總是應用此變換。
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                # 隨機調整亮度和對比度。p=1 表示總是應用此變換。
                albu.RandomBrightnessContrast(p=1),
                
                # 隨機調整圖像的色相、飽和度和亮度。p=1 表示總是應用此變換。
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ])