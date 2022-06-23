certer = (6, 6)

human_start_nl = (3.5, 12)
human_start_nr = (8.5, 12)
human_start_sl = (3.5, 0)
human_start_sr = (8.5, 0)
human_start_wu = (0, 8.5)
human_start_wd = (0, 3.5)
human_start_eu = (12, 8.5)
human_start_ed = (12, 3.5)
human_wp_nw = (3.5, 8.5)
human_wp_ne = (8.5, 8.5)
human_wp_sw = (3.5, 3.5)
human_wp_se = (8.5, 3.5)

car_start_n = (5, 12)
car_start_s = (7, 0)
car_start_w = (0, 5)
car_start_e = (12, 7)
car_wp_nw = (5, 7)
car_wp_ne = (7, 7)
car_wp_sw = (5, 5)
car_wp_se = (7, 5)
car_end_n = (7, 12)
car_end_s = (5, 0)
car_end_w = (0, 7)
car_end_e = (12, 5)

def car_path(start_dir:str, choice:str):
    assert(start_dir in ['n','s','w','e']), (f'Start direction [{start_dir}] not found. Must be "n/s/w/e".')
    assert(choice in ['l','s','r']), (f'Action choice [{choice}] not found. Must be "l/s/r".')
    if   start_dir == 'n':
        if choice == 'l':
            path = [car_start_n, car_wp_sw, car_end_e]
        elif choice == 's':
            path = [car_start_n, car_end_s]
        else:
            path = [car_start_n, car_wp_nw, car_end_w]
    elif start_dir == 's':
        if choice == 'l':
            path = [car_start_s, car_wp_ne, car_end_w]
        elif choice == 's':
            path = [car_start_s, car_end_n]
        else:
            path = [car_start_s, car_wp_se, car_end_e]
    elif start_dir == 'w':
        if choice == 'l':
            path = [car_start_w, car_wp_se, car_end_n]
        elif choice == 's':
            path = [car_start_w, car_end_e]
        else:
            path = [car_start_w, car_wp_sw, car_end_s]
    else: #start_dir == 'e'
        if choice == 'l':
            path = [car_start_e, car_wp_nw, car_end_s]
        elif choice == 's':
            path = [car_start_e, car_end_w]
        else:
            path = [car_start_e, car_wp_ne, car_end_n]
    return path

def human_path(start_dir:str, index:int):
    assert(start_dir in ['nl','nr','sl','sr','wu','wd','eu','ed']), (f'Start direction [{start_dir}] not found. Must be "n/s/w/e + l/r/u/d".')
    assert(0<index<10), ('Index must be in [1,9].')
    path_dict_nl = {1: [human_start_nl, human_wp_nw, human_start_wu], #//
                    2: [human_start_nl, human_wp_nw, human_wp_sw, human_start_wd],
                    3: [human_start_nl, human_wp_nw, human_wp_sw, human_start_sl],
                    4: [human_start_nl, human_wp_nw, human_wp_sw, human_wp_se, human_start_ed],
                    5: [human_start_nl, human_wp_nw, human_wp_sw, human_wp_se, human_start_sr], #//
                    6: [human_start_nl, human_wp_nw, human_wp_ne, human_start_nr],
                    7: [human_start_nl, human_wp_nw, human_wp_ne, human_start_eu],
                    8: [human_start_nl, human_wp_nw, human_wp_ne, human_wp_se, human_start_ed],
                    9: [human_start_nl, human_wp_nw, human_wp_ne, human_wp_se, human_start_sr],}

    path_dict_nr = {1: [human_start_nr, human_wp_ne, human_start_eu], #//
                    2: [human_start_nr, human_wp_ne, human_wp_nw, human_start_nl],
                    3: [human_start_nr, human_wp_ne, human_wp_nw, human_start_wu],
                    4: [human_start_nr, human_wp_ne, human_wp_nw, human_wp_sw, human_start_wd],
                    5: [human_start_nr, human_wp_ne, human_wp_nw, human_wp_sw, human_start_sl], #//
                    6: [human_start_nr, human_wp_ne, human_wp_se, human_start_ed],
                    7: [human_start_nr, human_wp_ne, human_wp_se, human_start_sr],
                    8: [human_start_nr, human_wp_ne, human_wp_se, human_wp_sw, human_start_wd],
                    9: [human_start_nr, human_wp_ne, human_wp_se, human_wp_sw, human_start_sl],}

    path_dict_sl = {1: [human_start_sl, human_wp_sw, human_start_wd], #//
                    2: [human_start_sl, human_wp_sw, human_wp_nw, human_start_nl],
                    3: [human_start_sl, human_wp_sw, human_wp_nw, human_start_wu],
                    4: [human_start_sl, human_wp_sw, human_wp_nw, human_wp_ne, human_start_nr],
                    5: [human_start_sl, human_wp_sw, human_wp_nw, human_wp_ne, human_start_eu], #//
                    6: [human_start_sl, human_wp_sw, human_wp_se, human_start_ed],
                    7: [human_start_sl, human_wp_sw, human_wp_se, human_start_sr],
                    8: [human_start_sl, human_wp_sw, human_wp_se, human_wp_ne, human_start_nr],
                    9: [human_start_sl, human_wp_sw, human_wp_se, human_wp_ne, human_start_eu],}
    
    path_dict_sr = {1: [human_start_sr, human_wp_se, human_start_ed], #//
                    2: [human_start_sr, human_wp_se, human_wp_ne, human_start_nr],
                    3: [human_start_sr, human_wp_se, human_wp_ne, human_start_eu],
                    4: [human_start_sr, human_wp_se, human_wp_ne, human_wp_nw, human_start_nl],
                    5: [human_start_sr, human_wp_se, human_wp_ne, human_wp_nw, human_start_wu], #//
                    6: [human_start_sr, human_wp_se, human_wp_sw, human_start_wd],
                    7: [human_start_sr, human_wp_se, human_wp_sw, human_start_sl],
                    8: [human_start_sr, human_wp_se, human_wp_sw, human_wp_nw, human_start_nl],
                    9: [human_start_sr, human_wp_se, human_wp_sw, human_wp_nw, human_start_wu],}

    path_dict_wu = {1: [human_start_wu, human_wp_nw, human_start_nl], #//
                    2: [human_start_wu, human_wp_nw, human_wp_sw, human_start_wd],
                    3: [human_start_wu, human_wp_nw, human_wp_sw, human_start_sl],
                    4: [human_start_wu, human_wp_nw, human_wp_sw, human_wp_se, human_start_ed],
                    5: [human_start_wu, human_wp_nw, human_wp_sw, human_wp_se, human_start_sr], #//
                    6: [human_start_wu, human_wp_nw, human_wp_ne, human_start_nr],
                    7: [human_start_wu, human_wp_nw, human_wp_ne, human_start_eu],
                    8: [human_start_wu, human_wp_nw, human_wp_ne, human_wp_se, human_start_ed],
                    9: [human_start_wu, human_wp_nw, human_wp_ne, human_wp_se, human_start_sr],}

    path_dict_wd = {1: [human_start_wd, human_wp_sw, human_start_sl], #//
                    2: [human_start_wd, human_wp_sw, human_wp_nw, human_start_nl],
                    3: [human_start_wd, human_wp_sw, human_wp_nw, human_start_wu],
                    4: [human_start_wd, human_wp_sw, human_wp_nw, human_wp_ne, human_start_nr],
                    5: [human_start_wd, human_wp_sw, human_wp_nw, human_wp_ne, human_start_eu], #//
                    6: [human_start_wd, human_wp_sw, human_wp_se, human_start_ed],
                    7: [human_start_wd, human_wp_sw, human_wp_se, human_start_sr],
                    8: [human_start_wd, human_wp_sw, human_wp_se, human_wp_ne, human_start_nr],
                    9: [human_start_wd, human_wp_sw, human_wp_se, human_wp_ne, human_start_eu],}

    path_dict_eu = {1: [human_start_eu, human_wp_ne, human_start_nr], #//
                    2: [human_start_eu, human_wp_ne, human_wp_nw, human_start_nl],
                    3: [human_start_eu, human_wp_ne, human_wp_nw, human_start_wu],
                    4: [human_start_eu, human_wp_ne, human_wp_nw, human_wp_sw, human_start_wd],
                    5: [human_start_eu, human_wp_ne, human_wp_nw, human_wp_sw, human_start_sl], #//
                    6: [human_start_eu, human_wp_ne, human_wp_se, human_start_ed],
                    7: [human_start_eu, human_wp_ne, human_wp_se, human_start_sr],
                    8: [human_start_eu, human_wp_ne, human_wp_se, human_wp_sw, human_start_wd],
                    9: [human_start_eu, human_wp_ne, human_wp_se, human_wp_sw, human_start_sl],}

    path_dict_ed = {1: [human_start_ed, human_wp_se, human_start_sr], #//
                    2: [human_start_ed, human_wp_se, human_wp_ne, human_start_nr],
                    3: [human_start_ed, human_wp_se, human_wp_ne, human_start_eu],
                    4: [human_start_ed, human_wp_se, human_wp_ne, human_wp_nw, human_start_nl],
                    5: [human_start_ed, human_wp_se, human_wp_ne, human_wp_nw, human_start_wu], #//
                    6: [human_start_ed, human_wp_se, human_wp_sw, human_start_wd],
                    7: [human_start_ed, human_wp_se, human_wp_sw, human_start_sl],
                    8: [human_start_ed, human_wp_se, human_wp_sw, human_wp_nw, human_start_nl],
                    9: [human_start_ed, human_wp_se, human_wp_sw, human_wp_nw, human_start_wu],}

    path_dict_dict = {'nl':path_dict_nl, 'nr':path_dict_nr, 'sl':path_dict_sl, 'sr':path_dict_sr,
                      'eu':path_dict_eu, 'ed':path_dict_ed, 'wu':path_dict_wu, 'wd':path_dict_wd}
    path = path_dict_dict[start_dir][index]

    return path