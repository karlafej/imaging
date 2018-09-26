datapath = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/"
tb_viz_path = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/logs/tb_logs"
tb_logs_path = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/logs/tb_viz"
modelpath = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/models/"
predspath = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/ppp/"
#mods = ["st_start", "st_middle", "st_end"]
#mods = ["start", "middle", "male_end", "female_end", "st_start", "st_middle", "st_male_end"] #None
mods = ["st_male_end"]
#mods = ["male_end"]
#mods = ["female_end"]
epochs = 6
modelfiles = {'middle': "model_middle_2018-09-25_19h18",
              'start': "model_start_2018-09-25_15h59",
              'end': "model_end_2018-04-04_22h08",
              'female_end': "model_female_end_2018-09-26_05h42",
              'male_end': "model_male_end_2018-09-26_02h36",
              'st_start': "model_st_start_2018-09-26_08h08",
              'st_middle': "model_st_middle_2018-09-26_08h34",
              'st_end': "model_st_end_2018-04-27_02h38",
              'st_female_end': "model_st_end_2018-04-27_02h38"
              'st_male_end' : "model_st_male_end_2018-09-26_10h53"}

