import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import Vanilla_NN, MFVI_NN

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, train_info = None):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])
    all_acc_for_save = np.zeros((data_gen.max_iter, data_gen.max_iter), dtype=np.float32)
    
    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            print('Vanilla NN train for task 0!')
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_variances = None
            ml_model.close_session()

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
        print('Current task : {}'.format(task_id))
        # Train on non-coreset data
        mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
        all_acc = utils.concatenate_results(acc, all_acc)
        
        
        for u in range(task_id + 1):

            print('>>> Test on task {:2d} : acc={:5.1f}% <<<'.format(u, 100 * acc[u]))
            all_acc_for_save[task_id, u] = acc[u]

        # Save

        log_name = '{}_{}_{}_{}epochs_batch{}_{}_{}coreset_{}'.format(train_info['date'], train_info['experiment'], train_info['tasknum'], no_epochs, train_info['batch'], train_info['coreset_method'], coreset_size, train_info['trial'])
        save_path = './results/' + log_name + '.txt'
        print('Save at ' + save_path)
        np.savetxt(save_path, all_acc_for_save, '%.4f')
        
        
        
        mf_model.close_session()

    return all_acc
