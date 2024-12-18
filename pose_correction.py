
def pose_correction (keypoints_data, predicted_class) :

    keypoints = keypoints_data[0]['keypoints']
    
    keypoints_sorted_x = sorted(keypoints, key=lambda k: k['x'])
    keypoints_sorted_y = sorted(keypoints, key=lambda k : -k['y'])
    
    #plank
    plank_keypoint0_x = keypoints_sorted_x[0]['x']
    plank_keypoint0_y = keypoints_sorted_x[0]['y']
    plank_keypoint2_x = keypoints_sorted_x[2]['x']
    plank_keypoint2_y = keypoints_sorted_x[2]['y']
    plank_keypoint4_x = keypoints_sorted_x[4]['x']
    plank_keypoint4_y = keypoints_sorted_x[4]['y']
    
    #squat
    squat_keypoint0_x = keypoints_sorted_y[0]['x']
    squat_keypoint2_x = keypoints_sorted_y[2]['x']
    
    #warrior2
    warrior2_keypoint0_y = keypoints_sorted_x[0]['y']
    warrior2_keypoint1_y = keypoints_sorted_x[1]['y']
    
    left_val = min(warrior2_keypoint0_y,warrior2_keypoint1_y)
    
    warrior2_keypoint00_y = keypoints_sorted_x[-1]['y']
    warrior2_keypoint11_y = keypoints_sorted_x[-2]['y']
    right_val = min(warrior2_keypoint00_y,warrior2_keypoint11_y)
    

    
    print('===========sort=============')
    for keypoint in keypoints_sorted_x:
    	print(f"{keypoint['body_part']} : {keypoint['x']} , {keypoint['y']}")
    	
    print('============================')
    	

    for keypoint in keypoints :
        if keypoint['body_part'] == 'RHip' :
            RHip_x = keypoint['x']
            RHip_y = keypoint['y']
            print(f'RHip : {RHip_x} , {RHip_y}')

        elif keypoint['body_part'] == 'RShoulder' :
            RShoulder_x = keypoint['x']
            RShoulder_y = keypoint['y']
            print(f'RShoulder : {RShoulder_x} , {RShoulder_y}')

        elif keypoint['body_part'] == 'RElbow' :
            RElbow_x = keypoint['x']
            RElbow_y = keypoint['y']
            print(f'RElbow : {RElbow_x} , {RElbow_y}')

        elif keypoint['body_part'] == 'RAnkle' :
            RAnkle_x = keypoint['x']
            RAnkle_y = keypoint['y']
            print(f'RAnkle : {RAnkle_x} , {RAnkle_y}')

        elif keypoint['body_part'] == 'RKnee' :
            RKnee_x = keypoint['x']
            RKnee_y = keypoint['y']
            print(f'RKnee : {RKnee_x} , {RKnee_y}')


    print(f'pose_correction: {keypoints}')
    if predicted_class == -1 :
        print('No Classification')
        return 'None'

    elif predicted_class == 0 :
        if plank_keypoint0_y<plank_keypoint2_y or abs(plank_keypoint4_y - plank_keypoint0_y) >50 :
            return 'Plank : Keep your shoulders and hips in a straight line.'
        else :
            return 'Plank : good pose!!'

    elif predicted_class == 1:
        if abs(squat_keypoint0_x-squat_keypoint2_x) > 55 :
            return 'Squat : Don\'t let your knees go past your toes.'

        else :
            return 'Squat : good pose!!'

    elif predicted_class == 2 :
        if abs(left_val-right_val) >= 30 :
            return 'Warrior2 : Keep your shoulders and Elbow in a straight line.'
        else :
            return 'Warrior2 : good pose!!'

    else :
        return "error"
