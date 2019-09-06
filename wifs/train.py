from CGvsPhoto import Model

model = Model(database_path = './newpatch', image_size = 100,
              filters = [32, 64],
              feature_extractor = 'Stats', batch_size = 50,
              using_GPU = True, only_green = True)

model.train(nb_train_batch = 15000,
            nb_test_batch = 80,
            nb_validation_batch = 40)

test_data_path = './target222/test/'

model.test_total_images(test_data_path = test_data_path,
                      nb_images = 720, decision_rule = 'weighted_vote')

