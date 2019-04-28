def generate_simdata(db,resolution,pyramid_height,pyramid_width,source_position,polarisation, frequency):
    
    current_conf=[
    {
        "simulate": {
            "resolution": resolution,
            "use_fixed_time": false,
            "simulation_time": 30,
            "dpml": 0.1,
            "padding": 0.1,
            "ff_pts": 1600,
            "ff_cover": false,
            "use_symmetries": true,
            "calculate_flux": true,
            "ff_calculations": true,
            "ff_angle": 6,
            "simulation_ratio": "6/5",
            "substrate_ratio": "1/20"
        },
        "pyramid": {
            "source_position": 0.06,
            "pyramid_height": 3.1,
            "pyramid_width": 2,
            "source_direction": "mp.Ey",
            "frequency_center": 2,
            "frequency_width": 0.5,
            "number_of_freqs": 1,
            "cutoff": 2
        },
        "result": {}
    }
]
    write(db,current_conf)