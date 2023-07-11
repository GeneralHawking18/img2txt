from base_cap_models import blip

def create(args):
    if args.base_model == "blip1":
        return blip.BLIP_1()

def gen_init_cap(model, image):
    return model(image)
    

"""def gen_init_cap(args, image):
    if args.base_model == "blip1":
        model = blip.BLIP_1()
        caption = model(image)
    return caption
    """