

def get_crop_window(params):

    final_height = params.image_height - (params.crop_up + params.crop_down)
    final_width = params.image_width

    return [
        params.crop_up,
        0,
        final_height,
        final_width,
    ]