
def pose_correction (keypoints_data, predicted_class) :

    keypoints = keypoints_data[0]['keypoints']

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
        if abs(RShoulder_y - RHip_y) >= 150 :
            return 'Keep your shoulders and hips in a straight line.'
        else :
            return 'good pose!!'

    elif predicted_class == 1:
        if abs(RAnkle_x-RKnee_x) > 50 :
            return 'Don\'t let your knees go past your toes.'

        else :
            return 'good pose!!'

    elif predicted_class == 2 :
        if abs(RShoulder_y - RElbow_y) >= 30 :
            return 'Keep your shoulders and Elbow in a straight line.'
        else :
            return 'good pose!!'

    else :
        return "error"