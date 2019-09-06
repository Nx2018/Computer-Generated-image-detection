from CGvsPhoto import Database_loader

path_source = '/home/forensics/CGPG_experiment/WIFS/target/'
path_export = '/home/forensics/CGPG_experiment/WIFS/newpatch/'
size_patch = 100

data = Database_loader(path_source, size = size_patch,
                     only_green = True)

# export a patch database
data.export_database(path_export,
                     nb_train = 0,
                     nb_test = 0,
                     nb_validation = 1000)
