import logging
from config.configs import get_config
from src.train import Trainer
from src.translate import translate

def main(parser, usage_mode):
    if usage_mode == 'train':
        logging.info('#################################################')
        logging.info('You have entered train mode.')
        logging.info('#################################################')
        
        # train with Trainer
        trainer = Trainer(parser)
        trainer.train_test()

    elif usage_mode == 'translate' or usage_mode == 'evaluate' or usage_mode == 'inference':
        logging.info('#################################################')
        logging.info('You have entered inference mode.')
        logging.info('#################################################')
        
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
    # set logging
    logging.basicConfig(level=logging.INFO)

    logging.info('#################################################')
    logging.info('You have entered __main__.')
    logging.info('#################################################')
    
    # define ArgumentParser
    parser = get_config()

    # get user input of whether purpose is train or inference
    usage_mode = input('Enter the mode you want to use :')

    # run main
    main(parser, usage_mode)

    logging.info('#################################################')
    logging.info('You are exiting __main__.')
    logging.info('#################################################')