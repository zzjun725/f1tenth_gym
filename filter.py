import numpy as np

def filter_data(data, controls_data=None, lower_bound=None, upper_bound=None):
    # TODO mask1 = data[:, 4] > 0.8len(data) # 0.6 fric has one abnormal spike

    if lower_bound is None:
        mask_ = data[:, 4] > upper_bound
    else:
        mask1 = data[:, 4] > upper_bound
        mask2 = data[:, 4] < lower_bound
        mask_ = mask1 | mask2

    # 7.7: 0.3|0.6 - 0.32|0.8 - none|1.0 - (-0.4, 0.4)|1.2 - (-0.6, 0.6)|1.4
    
    control_length = 20
    mask_ = np.floor_divide(np.where(mask_)[0], control_length) * control_length
    mask_ = np.unique(mask_)

    filtered_array = [i for value in mask_ for i in range(value, value + 20)]
    filtered_array = np.array(filtered_array)

    filtered_array = np.delete(filtered_array, np.where(filtered_array>(len(data)-1))[0])
    print('filtered numpy: ', filtered_array.shape)

    if controls_data is None:
        if len(filtered_array) > 0:
            filtered_data = np.delete(data, filtered_array, axis=0)
        else:
            filtered_data = data
        return filtered_data
    else:
        if len(filtered_array) > 0:
            filtered_data = np.delete(data, filtered_array, axis=0)
            filtered_controls_data = np.delete(controls_data, filtered_array, axis=0)
        else:
            filtered_data = data
            filtered_controls_data = controls_data
        return filtered_data, filtered_controls_data
    

def main():
    file_name_ = 'data/Random_Spielberg_raceline/'

    friction = 0.6
    states = np.load(file_name_+'states_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    controls = np.load(file_name_+'controls_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    filtered_states, filtered_controls = filter_data(states, controls_data=controls, lower_bound=None, upper_bound=0.3)
    
    np.save(file_name_+'states_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_states)
    np.save(file_name_+'controls_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_controls)


    friction = 0.8
    states = np.load(file_name_+'states_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    controls = np.load(file_name_+'controls_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    filtered_states, filtered_controls = filter_data(states, controls_data=controls, lower_bound=None, upper_bound=0.32)
    
    np.save(file_name_+'states_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_states)
    np.save(file_name_+'controls_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_controls)


    friction = 1.2
    states = np.load(file_name_+'states_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    controls = np.load(file_name_+'controls_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    filtered_states, filtered_controls = filter_data(states, controls_data=controls, lower_bound=-0.4, upper_bound=0.4)
    
    np.save(file_name_+'states_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_states)
    np.save(file_name_+'controls_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_controls)


    friction = 1.4
    states = np.load(file_name_+'states_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    controls = np.load(file_name_+'controls_mb_fric_' + str(int(10 * friction)) + '_vel_77.npy')
    filtered_states, filtered_controls = filter_data(states, controls_data=controls, lower_bound=-0.6, upper_bound=0.6)
    
    np.save(file_name_+'states_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_states)
    np.save(file_name_+'controls_mb_fric_{}_vel_{}.npy'.format(int(10 * friction), 77), filtered_controls)


    
if __name__ == '__main__':
    main()