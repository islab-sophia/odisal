import train
import train_ms
import estimate
import estimate_ms
import evaluate

import argparse
from datetime import datetime
import os

import yaml

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/sample.yaml',
                        help='config file')
    return parser

def main(cfg, cfg_path):
    start_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
    
    estimate_resume = None
    salmap_path = None
    evaluate_log_path = None
    train_log_path = None
    
    if 'TRAIN' in cfg.keys():
        print("====TRAIN START====")
        if cfg['TRAIN']['MODEL']['USE_MULTISCALE_MODEL']:
            print("====USE MULTISCALE MODEL====")
            estimate_resume, train_log_path = train_ms.main(cfg['TRAIN'])
        else:
            estimate_resume, train_log_path = train.main(cfg['TRAIN'])
        print("====TRAIN FINISH====")

    print("estimate_resume: " + estimate_resume)
        
    if 'ESTIMATE' in cfg.keys():
        print("====ESTIMATE START====")
        if estimate_resume == None:
            estimate_resume = cfg['ESTIMATE']['SETTING']['RESUME']
        else:
            cfg['ESTIMATE']['SETTING']['RESUME'] = estimate_resume
            
        if cfg['ESTIMATE']['MODEL']['USE_MULTISCALE_MODEL']:
            print("====USE MULTISCALE MODEL====")
            salmap_path = estimate_ms.main(cfg['ESTIMATE'])
        else:
            if type(cfg['ESTIMATE']['MODEL']['VIEW_ANGLE']) is list:
                view_angle_list = cfg['ESTIMATE']['MODEL']['VIEW_ANGLE']
                salmap_path = []
                for view_angle in view_angle_list:
                    cfg['ESTIMATE']['MODEL']['VIEW_ANGLE'] = view_angle
                    salmap_path.append(estimate.main(cfg['ESTIMATE']))
            else:
                salmap_path = estimate.main(cfg['ESTIMATE'])
        print("====ESTIMATE FINISH====")
        
    if 'EVALUATE' in cfg.keys():
        print("====EVALUATE START====")
        if salmap_path == None:
            salmap_path = cfg['EVALUATE']['DATA']['SALMAP_PATH']
            evaluate_log_path = evaluate.main(cfg['EVALUATE'])
        else:
            if type(salmap_path) is str:
                cfg['EVALUATE']['DATA']['SALMAP_PATH'] = salmap_path
                evaluate_log_path = evaluate.main(cfg['EVALUATE'])
            else:
                evaluate_log_path = []
                for i in range(len(view_angle_list)):
                    cfg['EVALUATE']['DATA']['SALMAP_PATH'] = salmap_path[i]
                    evaluate_log_path.append(evaluate.main(cfg['EVALUATE'], view_angle=view_angle_list[i]))
        print("====EVALUATE FINISH====")

    finish_time_stamp = '{0:%Y%m%d-%H%M%S}'.format(datetime.now())
    print("FINISH TIME : {}".format(finish_time_stamp))

    log = {'start_time_stamp': start_time_stamp,
        'finish_time_stamp': finish_time_stamp,
        'estimate_resume': estimate_resume,
        'salmap_path': salmap_path,
        'evaluate_log_path': evaluate_log_path}

    pipeline_log_path = "{}_log.yaml".format(os.path.splitext(cfg_path)[0])
    with open(pipeline_log_path, 'w') as f:
        f.write(yaml.dump(log, default_flow_style=False))
    print("SAVE LOG : {}".format(pipeline_log_path))
    
if __name__ == '__main__':
    args = get_parser().parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.cfg)
