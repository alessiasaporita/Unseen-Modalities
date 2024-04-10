

def get_audio_configs(target_length=128):
    norm_stats = [-5.388, 2.94]

    train_config = {
        "num_mel_bins": 128,
        "target_length": target_length,
        "freqm": 48, #frequency masking
        "timem": 192, #time masking
        "mixup": 0.5, #never used
        "dataset": "Kinetics400", #never used
        "mode": "train",
        "mean": norm_stats[0], 
        "std": norm_stats[1],
        "noise": True,
    }
    val_config = {
        "num_mel_bins": 128,
        "target_length": target_length,
        "freqm": 0, #frequency masking, not do for eval set
        "timem": 0, #time masking, not do for eval set
        "mixup": 0, #never used
        "dataset": "Kinetics400",
        "mode": "evaluation",
        "mean": norm_stats[0],
        "std": norm_stats[1],
        "noise": False,
    }

    return train_config, val_config