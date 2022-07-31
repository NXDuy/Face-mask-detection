def parse_params(file_dir: str = "hyp/hyp-mask-face.yaml") -> dict:
    '''
    parse_params method is used to extract the training information in file hyp-mask-face (file_dir)
    Some contraints must be satisfy:
        - INTEGER value: batch_size, picture_spatial_size
        - STR value: train, test, val, optimizer
        - Other parameter should be float value
    '''
    params = {}

    INT_PARAMS = ["batch_size", "picture_spatial_size"]
    STR_PARAMS = ["train", "test", "val", "optimizer"]

    with open(file_dir) as f:
        for model_params in f:
            model_params = model_params.strip()
            if model_params is None or len(model_params) == 0:
                continue

            param_info = model_params.split(":")
            param_key = param_info[0].strip()
            param_value = param_info[1].strip()

            if param_key in INT_PARAMS:
                try:
                    param_value = int(param_value)
                    params[param_key] = param_value
                    continue
                except:
                    raise ValueError("\033[1;32;41m %s SHOULD BE INTEGER"%(param_key))

            if param_key in STR_PARAMS:
                params[param_key] = param_value
            else:
                try:
                    param_value = float(param_value)
                    params[param_key] = param_value
                except:
                    raise ValueError("\033[1;32;41m %s SHOULD BE FLOAT"%(param_key))

    return params


