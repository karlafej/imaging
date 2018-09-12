datapath = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/"
tb_viz_path = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/logs/tb_logs"
tb_logs_path = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/logs/tb_viz"
modelpath = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/models/"
predspath = "/home/fejfark/PRIMUS/data/18_lab/karla/imaging/dl/ppp/"
#mods = ["st_start", "st_middle", "st_end"]
mods = ["start", "middle", "male_end", "female_end"] #None
#mods = ["male_end"]
#mods = ["female_end"]
epochs = 6
modelfiles = {'middle': "model_middle_2018-08-03_18h29",
              'start': "model_start_2018-08-03_16h30",
              'end': "model_end_2018-04-04_22h08",
              'female_end': "model_female_end_2018-08-06_16h40",
              'male_end': "model_male_end_2018-08-03_22h49",
              'st_start': "model_st_start_2018-04-26_15h32",
              'st_middle': "model_st_middle_2018-04-26_18h10",
              'st_end': "model_st_end_2018-04-27_02h38"}
