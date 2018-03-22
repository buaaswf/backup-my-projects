import os
import sys
import numpy as np

from PIL import Image

import net
from utils import depth_montage, normals_montage
#from keras.optimizers import SGD

def main():
    # location of depth module, config and parameters

    model_name = 'depthnormals_nyud_alexnet'
    #model_name = 'depthnormals_nyud_vgg'

#    module_fn = 'models/iccv15/%s.py' % model_name
#    config_fn = 'models/iccv15/%s.conf' % model_name
#    params_dir = 'weights/iccv15/%s' % model_name

    module_fn = 'models/iccv15/depthnormals_nyud_alexnet.py' #% model_name
    config_fn = 'models/iccv15/depthnormals_nyud_alexnet.conf' #% model_name
    params_dir = 'weights/iccv15/depthnormals_nyud_alexnet' #% model_name
    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    # demo image
    rgb = Image.open('thyroid.jpg')
    rgb = rgb.resize((320, 240), Image.BICUBIC)
    #rergb = Image.fromarray(rgb)
    rgb.save("thyroid.jpg")

    # build depth inference function and run
    rgb_imgs = np.asarray(rgb).reshape((1, 240, 320, 3))
    (pred_depths, pred_normals) = machine.infer_depth_and_normals(rgb_imgs)

    # save prediction
    depth_img_np = depth_montage(pred_depths)
    depth_img = Image.fromarray((255*depth_img_np).astype(np.uint8))
    depth_img.save('demo_depth_prediction_thyroid.png')

    normals_img_np = normals_montage(pred_normals)
    normals_img = Image.fromarray((255*normals_img_np).astype(np.uint8))
    normals_img.save('demo_normals_prediction_thyroid.png')
def machine2model():
    module_fn = 'models/iccv15/depthnormals_nyud_alexnet.py' #% model_name
    config_fn = 'models/iccv15/depthnormals_nyud_alexnet.conf' #% model_name
    params_dir = 'weights/iccv15/depthnormals_nyud_alexnet' #% model_name
    trainMachine = net.create_machine(module_fn, config_fn, params_dir)
    #model = trainMachine.define_machine()
    print trainMachine
    return trainMachine
def traindepth(data = None):
    train_set_x, train_set_y, train_set_m = datasets[0] 
    valid_set_x, valid_set_y, train_set_m = datasets[1] 
    test_set_x, test_set_y, train_set_m = datasets[2] 
    index = T.lscalar()
    x = T.matrix ('x')
    y = T.ivector ('y0')
    m = T.,matrix('m0')
    #traindata, testdata = data('train','test')
    module_fn = 'models/iccv15/depthnormals_nyud_alexnet.py' #% model_name
    config_fn = 'models/iccv15/depthnormals_nyud_alexnet.conf' #% model_name
    params_dir = 'weights/iccv15/depthnormals_nyud_alexnet' #% model_name
    #trainMachine = net.create_machine(module_fn, config_fn, params_dir)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model = Sequential()
    model = machine2model()
    print "======>>", model
    pred = machine.infer_depth_and_normals(images)
    cost = machine.define_depths_cost(pred, y0, m0)
    for u in model:
	updatedic = u.get_update(cost, machine.conf.geteval(l0.name,'learning_rate_scale'), machine.config.geteval('train','momentum'))
        updates += updatedict
    print type(updates)
    train_model = theano.fucntion
	(
	inputs = [index],
	outputs = cost,
 	updates=updates,
	givens={
	   x:train_set_x[index * batch_size: (index + 1) * batch_size],
	   y: train_set_y[index * batch_size: (index + 1) * batch_size],
	   m: train_set_m[index * batch_size: (index + 1) * batch_size]
		}
	)


    #############
    #train model#
    #############
    #model.compile(loss=loss,optimizer = sgd, metrics=['accuracy'])
    #model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (x_test, y_test), shuffle = True)
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    

if __name__ == '__main__':
    #main()
    traindepth('')

