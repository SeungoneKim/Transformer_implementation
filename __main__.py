import logging
import sys
from config.configs import get_config
from src.train import Trainer
from src.translate import translate

def main(parser, usage_mode):

    if usage_mode == 'train':
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have entered train mode.\n')
        sys.stdout.write('#################################################\n')
        
        # train with Trainer
        trainer = Trainer(parser)
        trainer.train_test()

    elif usage_mode == 'translate' or usage_mode == 'evaluate' or usage_mode == 'inference':
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have entered inference mode.\n')
        sys.stdout.write('#################################################\n')
        
        while True:
            # get user input
            user_input = input('Enter the sentence you with to convert : ')
            
            # run translate
            result = translate(user_input)
            print(result)

            continue_translate = input('Will you proceed? : ')
            if continue_translate != 'yes':
                break
    
    else:
        assert "You have gave wrong mode"

if __name__ == "__main__":

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You have entered __main__.\n')
    sys.stdout.write('#################################################\n')
    
    # define ArgumentParser
    parser = get_config()

    # get user input of whether purpose is train or inference
    usage_mode = input('Enter the mode you want to use :')

    # run main
    main(parser, usage_mode)

    sys.stdout.write('#################################################\n')
    sys.stdout.write('You are exiting __main__.\n')
    sys.stdout.write('#################################################\n')