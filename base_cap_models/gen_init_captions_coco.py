from gen_utils import generate_caption
import json
import os


def generate_init_cap(model_name, model, processor, device, image_instance = None, image = None):
    if image_instance:
        image = processor(image_instance).unsqueeze(0).to(device)

    if model_name == "blip-1":
        init_cap = model.generate({"image": image})
    elif model_name == "blip-2":
        init_cap = model.generate({"image": image})
    
    elif model_name == "cnn_lstm":
        pass

    return init_cap

def run_caption(args, img_name, img_pil_list, init_caption, lm_model, lm_tokenizer, clip, token_mask, logger, all_results):
    image_instance = img_pil_list
    gen_texts, clip_scores = generate_caption(
            img_name,
            lm_model,
            clip, lm_tokenizer,
            image_instance, init_caption,
            token_mask, logger,
            prompt=args.prompt, batch_size=args.batch_size, max_len=args.sentence_len,
            top_k=args.candidate_k, temperature=args.lm_temperature,
            max_iter=args.num_iterations,
            alpha=args.alpha,beta=args.beta,theta=args.theta,
            generate_order = args.order
          )

    """for iter_id, gen_text_list in enumerate(gen_texts):
        # for jj in range(len(gen_text_list)):
        for jj in [len(gen_text_list) - 1]:
            image_id = img_name[0] # img_name[jj].split(".")[0]
            gen_text_list[jj] = gen_text_list[jj].replace('image of ', '').capitalize()
            print("fix bug", all_results)
            if all_results[iter_id]==None:
                # all_results[iter_id] = {image_id: gen_text_list[jj]}
                all_results[iter_id] = []
            else:
                all_results[iter_id].append({"image_id": image_id,
                                         "caption": gen_text_list[jj]})"""
    
    # for jj in range(len(gen_text_list)):
    # for jj in [len(gen_text_list) - 1]:
    jj = len(gen_texts[0]) - 2
    image_id = img_name[0] # img_name[jj].split(".")[0]
    gen_texts[0][jj] = gen_texts[0][jj].replace('image of ', '').capitalize()
        
    all_results.append({"image_id": image_id,
                        "caption": gen_texts[0][jj]})
                            

    return all_results


def gen_coco_cap_json(
    args, run_type, 
    lm_model, lm_tokenizer, clip, token_mask, logger,
    train_loader, 
):
    if args.run_type == 'caption' and args.order == 'precise':
        for sample_id in range(args.samples_num):
            all_results = [] # [None] * (args.num_iterations+1)
            logger.info(f"Sample {sample_id+1}: ")
            for batch_idx, (img_batch_pil_list, name_batch_list, init_caption_list) in enumerate(train_loader):
                logger.info(f"The {batch_idx+1}-th batch:")
                all_results = run_caption(
                    args, name_batch_list, img_batch_pil_list,
                    init_caption_list,
                    lm_model,
                    lm_tokenizer, clip, token_mask, logger, all_results)
            
                save_dir = "results/caption_%s_len%d_topk%d_alpha%.3f_beta%.3f_gamma%.3f_theta%.3f_lmTemp%.3f/sample_%d" % (
                    args.order,args.sentence_len, args.candidate_k, args.alpha, args.beta,args.gamma,args.theta,args.lm_temperature,sample_id
                )
                if os.path.exists(save_dir) == False:
                    os.makedirs(save_dir)

                cur_json_file = os.path.join(save_dir,f"coco_caption.json")
                with open(cur_json_file,'w') as _json:
                    json.dump(all_results, _json)


