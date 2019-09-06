from CGvsPhoto import Database_loader

# directory with the original database
source_db = '/home/forensics/CGPG_experiment/WIFS/target/'

# wanted size for the patches 
image_size = 100

# directory to store the patch database
target_patches = '/home/forensics/CGPG_experiment/WIFS/newpatch/'


# create a database manager 
DB = Database_loader(source_db, image_size, 
                     only_green=False, rand_crop = False)

# export a patch database    
DB.export_database(target_patches, 
                   nb_train = 40000, 
                   nb_test = 2000, 
                   nb_validation = 1000)

# directory to store splicing images 
# target_splicing = '/home/nicolas/Database/splicing2/'


# DB.export_splicing(target_splicing, 50)
