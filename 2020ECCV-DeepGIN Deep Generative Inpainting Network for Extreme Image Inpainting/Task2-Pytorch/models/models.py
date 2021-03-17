import torch 

def create_model(opt):
    if opt.model == 'Ours': 
        from .our_model import OurModel, InferenceModel 
        if opt.isTrain: 
            model = OurModel()
        else:
            model = InferenceModel()
    else: 
        print('Please define your model [%s]!'.format(opt.model)) 
    model.initialize(opt)
    print("model [%s] was created" % (model.name())) 
    num_params_G, num_params_D = model.get_num_params() 

    if opt.isTrain and len(opt.gpu_ids): 
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids) 

    return model, num_params_G, num_params_D 

    