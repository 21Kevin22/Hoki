# Re-implemented data from 3D Scene Graph dataset
# Converting data structure from graph to nested Python dictionary
# Original data is in .npz format
# Added new objects for evaluation tasks (e.g., CPU, GPU, mop, banana_peel, locker, etc.)
# Source with MIT License: https://github.com/StanfordVL/3DSceneGraph


import copy


ALLENSVILLE = {
    "name": "allensville",
    "rooms": {
        "bathroom_1": {
            "items": {
                "psu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "sink_1": {
                    "accessible": True,
                    "affordance": ["clean_mop"],
                    "state": "free"
                },
                "toilet_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "plant_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "mop": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "clean_mop", "mop_floor"],
                    "state": "clean"
                }
            },
            "neighbor": ["corridor_2"]
        },
        "bathroom_2": {
            "items": {
                "gpu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "sink_2": {
                    "accessible": True,
                    "affordance": ["clean_mop"],
                    "state": "free"
                },
                "toilet_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "plant_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_3"]
        },
        "bedroom_1": {
            "items": {
                "mainboard": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "glass": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "bed_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "shelf": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "load", "unload"],
                    "state": "loaded",
                    "content": {
                        "book": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free"
                        }
                    }
                }
            },
            "neighbor": ["corridor_2"]
        },
        "bedroom_2": {
            "items": {
                "cpu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "rotting_apple": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "plate": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "bed_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "lamp": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_3"]
        },
        "corridor_1": {
            "items": {},
            "neighbor": ["lobby", "corridor_3"]
        },
        "corridor_2": {
            "items": {
                "fridge_1": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                },
                "fridge_2": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                }
            },
            "neighbor": ["bathroom_1", "bedroom_1", "corridor_3"]
        },
        "corridor_3": {
            "items": {},
            "neighbor": ["corridor_1", "corridor_2", "bathroom_2", "bedroom_2", "kitchen", "living_room"]
        },
        "dining_room": {
            "items": {
                "ssd": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "clock": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "cola_can": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "chair_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "chair_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "dining_table": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["kitchen", "living_room"]
        },
        "kitchen": {
            "items": {
                "knife": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "fork": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "spoon": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "microwave": {
                    "accessible": True,
                    "affordance": ["open", "close", "turnon", "turnoff"],
                    "state": "closed, off"
                },
                "oven": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "closed, off"
                },
                "rubbish_bin": {
                    "accessible": True,
                    "affordance": ["dispose"],
                    "state": "free"
                },
                "fridge_3": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "free"
                },
                "chair_3": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_3", "dining_room"]
        },
        "living_room": {
            "items": {
                "desk": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free"
                },
                "bowl_2": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "bowl_3": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "robot_hub": {
                    "accessible": True,
                    "affordance": ["charge"],
                    "state": "free"
                },
                "chair_4": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "chair_5": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "couch": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_3", "dining_room"]
        },
        "lobby": {
            "items": {
                "ram": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "banana_peel": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "flower": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "locker": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "load", "unload"],
                    "state": "loaded",
                    "content": {
                        "paper": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free"
                        }
                    }
                }
            },
            "neighbor": ["corridor_1"]
        }
    },
    "agent": {
        "position": "living_room",
        "state": "hand-free"
    },
}

KEMBLESVILLE = {
    "name": "kemblesville",
    "rooms": {
        "bathroom": {
            "items": {
                "sink_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "toilet": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "detergent": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_1", "closet_1"]
        },
        "closet_1": {
            "items": {
                "clothes": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                }
            },
            "neighbor": ["bathroom", "bedroom_1"]
        },
        "closet_2": {
            "items": {
                "fridge_1": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                },
                "fridge_2": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                }
            },
            "neighbor": ["corridor_2", "kitchen"]
        },
        "corridor_1": {
            "items": {},
            "neighbor": ["bedroom_1", "bedroom_2", "bathroom", "living_room"]
        },
        "corridor_2": {
            "items": {},
            "neighbor": ["closet_2", "kitchen", "living_room"]
        },
        "bedroom_1": {
            "items": {
                "bed_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_1", "closet_1"]
        },
        "bedroom_2": {
            "items": {
                "bed_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "book": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_1"]
        },
        "kitchen": {
            "items": {
                "wash_machine": {
                    "accessible": True,
                    "affordance": ["open", "close", "turnon", "turnoff"],
                    "state": "closed, off"
                },
                "microwave": {
                    "accessible": True,
                    "affordance": ["open", "close", "turnon", "turnoff"],
                    "state": "closed, off"
                },
                "oven": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "closed, off"
                },
                "sink_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "fridge_3": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                }
            },
            "neighbor": ["corridor_2", "closet_2"]
        },
        "living_room": {
            "items": {
                "tv": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "free"
                },
                "couch": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_1", "corridor_2"]
        },
    },
    "agent": {
        "position": "living_room",
        "state": "hand-free"
    },
}

PABLO = {
    "name": "pablo",
    "rooms": {
        "bathroom": {
            "items": {
                "sink": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "bottle": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "toilet": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor"]
        },
        "bedroom": {
            "items": {
                "book": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "vase": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "chair": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "bed": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["living_room"]
        },
        "closet": {
            "items": {},
            "neighbor": ["corridor"]
        },
        "corridor": {
            "items": {},
            "neighbor": ["bathroom", "closet", "living_room"]
        },
        "living_room": {
            "items": {
                "couch": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "tv": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "off"
                }
            },
            "neighbor": ["bedroom", "corridor"]
        }
    },
    "agent": {
        "position": "bedroom",
        "state": "hand-free"
    },
}

PAROLE = {
    "name": "parole",
    "rooms": {
        "bathroom": {
            "items": {
                "psu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "sink_1": {
                    "accessible": True,
                    "affordance": ["clean_mop"],
                    "state": "free"
                },
                "sink_2": {
                    "accessible": True,
                    "affordance": ["clean_mop"],
                    "state": "free"
                },
                "toilet": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "locker": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "load", "unload"],
                    "state": "loaded",
                    "content": {
                        "paper": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free"
                        }
                    }
                }
            },
            "neighbor": ["corridor_2"]
        },
        "bedroom": {
            "items": {
                "cpu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "rotting_apple": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "bed": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "glass": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "lamp": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "shelf": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "load", "unload"],
                    "state": "loaded",
                    "content": {
                        "book": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free"
                        }
                    }
                }
            },
            "neighbor": ["corridor_2"]
        },
        "dining_room": {
            "items": {
                "ram": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "cola_can": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "chair_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "chair_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "dining_table": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free"
                },
                "mug": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_1", "living_room"]
        },
        "corridor_1": {
            "items": {
                "flower": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "ssd": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "mop": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "clean_mop", "mop_floor"],
                    "state": "clean"
                }
            },
            "neighbor": ["corridor_2", "dining_room", "kitchen"]
        },
        "corridor_2": {
            "items": {
                "gpu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "banana_peel": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                }
            },
            "neighbor": ["bathroom", "bedroom", "corridor_1"]
        },
        "kitchen": {
            "items": {
                "mainboard": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "knife": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "fork": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "spoon": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "oven": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "closed, off"
                },
                "rubbish_bin": {
                    "accessible": True,
                    "affordance": ["dispose"],
                    "state": "free"
                },
                "fridge": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed"
                },
                "coffee_machine": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "off"
                }
            },
            "neighbor": ["corridor_1"]
        },
        "living_room": {
            "items": {
                "plate": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "couch_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "couch_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "robot_hub": {
                    "accessible": True,
                    "affordance": ["charge"],
                    "state": "free"
                },
                "desk": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["dining_room"]
        },
    },
    "agent": {
        "position": "living_room",
        "state": "hand-free"
    },
    "human": {
        "name": "Tom",
        "position": "bedroom",
        "state": ""
    },
}

ROSSER = {
    "name": "rosser",
    "rooms": {
        "bathroom": {
            "items": {
                "sink_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "sink_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "toilet": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["closet"]
        },
        "bedroom": {
            "items": {
                "bed": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "tv": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "free"
                }
            },
            "neighbor": ["closet", "home_office", "living_room"]
        },
        "closet": {
            "items": {},
            "neighbor": ["bathroom", "bedroom"]
        },
        "home_office": {
            "items": {
                "chair": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["bedroom", "living_room"]
        },
        "kitchen": {
            "items": {
                "microwave": {
                    "accessible": True,
                    "affordance": ["open", "close", "turnon", "turnoff"],
                    "state": "closed, off"
                },
                "oven": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "closed, off"
                },
                "sink_3": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "fridge": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                }
            },
            "neighbor": ["living_room"]
        },
        "living_room": {
            "items": {
                "couch": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["bedroom", "home_office", "kitchen"]
        }
    },
    "agent": {
        "position": "living_room",
        "state": "hand-free"
    },
}

SHELBIANA = {
    "name": "shelbiana",
    "rooms": {
        "bathroom_1": {
            "items": {
                "mainboard": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "sink_1": {
                    "accessible": True,
                    "affordance": ["clean_mop"],
                    "state": "free"
                },
                "toilet_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "mop": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "clean_mop", "mop_floor"],
                    "state": "clean"
                }
            },
            "neighbor": ["lobby"]
        },
        "bathroom_2": {
            "items": {
                "psu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "sink_2": {
                    "accessible": True,
                    "affordance": ["clean_mop"],
                    "state": "free"
                },
                "toilet_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_2"]
        },
        "balcony": {
            "items": {
                "dining_table": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["living_room", "bedroom_3"]
        },
        "closet": {
            "items": {
                "gpu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "chair": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "locker": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "load", "unload"],
                    "state": "loaded",
                    "content": {
                        "paper": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free"
                        }
                    }
                }
            },
            "neighbor": ["corridor_2"]
        },
        "corridor_1": {
            "items": {},
            "neighbor": ["corridor_2", "kitchen", "living_room"]
        },
        "corridor_2": {
            "items": {
                "glass": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                }
            },
            "neighbor": ["bathroom_2", "bedroom_2", "bedroom_3", "closet", "corridor_1"]
        },
        "living_room": {
            "items": {
                "cola_can": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "banana_peel": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "plate": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "couch": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "desk": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["balcony", "corridor_1", "lobby"]
        },
        "bedroom_1": {
            "items": {
                "cpu": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "bed_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                }
            },
            "neighbor": ["lobby"]
        },
        "bedroom_2": {
            "items": {
                "ssd": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "kite_1": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "rotting_apple": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "dispose"],
                    "state": "free"
                },
                "lamp": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_2"]
        },
        "bedroom_3": {
            "items": {
                "bed_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "kite_2": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "robot_hub": {
                    "accessible": True,
                    "affordance": ["charge"],
                    "state": "free"
                }
            },
            "neighbor": ["balcony", "corridor_2"]
        },
        "lobby": {
            "items": {
                "flower": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "clock": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "shelf": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "load", "unload"],
                    "state": "loaded",
                    "content": {
                        "book": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free"
                        }
                    }
                }
            },
            "neighbor": ["bathroom_1", "bedroom_1", "living_room"]
        },
        "kitchen": {
            "items": {
                "ram": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free"
                },
                "knife": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "fork": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "spoon": {
                    "accessible": True,
                    "affordance": ["pick", "drop", "place_on"],
                    "state": "free"
                },
                "microwave": {
                    "accessible": True,
                    "affordance": ["open", "close", "turnon", "turnoff"],
                    "state": "closed, off"
                },
                "oven": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "closed, off"
                },
                "sink_3": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free"
                },
                "fridge": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed, off"
                },
                "rubbish_bin": {
                    "accessible": True,
                    "affordance": ["dispose"],
                    "state": "free"
                }
            },
            "neighbor": ["corridor_1"]
        }
    },
    "agent": {
        "position": "bedroom_3",
        "state": "battery-full"
    },
}

OFFICE = {
    "name": "office",
    "rooms": {
        "peters_office": {
            "assets": {
                "cabinet_2": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "phone": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "apple_3": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "stapler_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    }
                },
                "desk_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                }
            },
            "neighbor": ["corridor_1"]
        },
        "tobis_office": {
            "assets": {
                "desk_38": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "pepsi": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_1"]
        },
        "meeting_room_1": {
            "assets": {
                "chair_3": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "chair_4": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "chair_5": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "table_5": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_2"]
        },
        "postdoc_bay_1": {
            "assets": {
                "desk_31": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "fire_extinguisher_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "desk_32": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "frame_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "frame_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "tshirt_7": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "tshirt_8": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_4"]
        },
        "phd_bay_1": {
            "assets": {
                "desk_7": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_8": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_9": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "tshirt_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "desk_10": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "tshirt_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "desk_11": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_12": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_6"]
        },
        "phd_bay_2": {
            "assets": {
                "desk_13": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_14": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_15": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "tshirt_3": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "desk_16": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_17": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_18": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "tshirt_4": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "tshirt_6": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_17"]
        },
        "admin": {
            "assets": {
                "shelf_1": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "fire_extinguisher_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
            },
            "neighbor": ["corridor_18"]
        },
        "printing_zone_2": {
            "assets": {
                "printer_2": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "off",
                    "items": {
                        "document": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_3"]
        },
        "meeting_room_2": {
            "assets": {
                "chair_2": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_5"]
        },
        "robot_lounge_2": {
            "assets": {},
            "neighbor": ["corridor_9"]
        },
        "robot_lounge_1": {
            "assets": {},
            "neighbor": ["corridor_3"]
        },
        "nikos_office": {
            "assets": {
                "chair_1": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "cabinet_1": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {},
                },
                "desk_1": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "coffee_mug": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_5"]
        },
        "michaels_office": {
            "assets": {
                "desk_6": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "scissorss": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "cabinet_6": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "stapler_3": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "poster": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
            },
            "neighbor": ["corridor_7"]
        },
        "aarons_office": {
            "assets": {},
            "neighbor": ["corridor_9"]
        },
        "jasons_office": {
            "assets": {
                "desk_5": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "monitor": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "cabine_5": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {},
                },
            },
            "neighbor": ["corridor_13"]
        },
        "mobile_robotics_lab": {
            "assets": {
                "table_4": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "book_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_8"]
        },
        "manipulation_lab": {
            "assets": {
                "table_3": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "book_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "gripper": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_15"]
        },
        "phd_bay_4": {
            "assets": {
                "desk_25": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "tshirt_5": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
                "desk_26": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_27": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_28": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_29": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_30": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_19"]
        },
        "postdoc_bay_3": {
            "assets": {
                "desk_35": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "doritos": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "desk_36": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "marker": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_21"]
        },
        "meeting_room_4": {
            "assets": {
                "table_2": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "janga": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                        "risk": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                        "monopoly": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_20"]
        },
        "filipes_office": {
            "assets": {
                "desk_37": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "stapler_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
            },
            "neighbor": ["corridor_10"]
        },
        "luis_office": {
            "assets": {},
            "neighbor": ["corridor_10"]
        },
        "wills_office": {
            "assets": {
                "desk_4": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "cabinet_4": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "apple_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "thesis_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "drone_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
            },
            "neighbor": ["corridor_10"]
        },
        "phd_bay_3": {
            "assets": {
                "desk_19": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_20": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_21": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "drone_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
                "desk_22": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_23": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
                "desk_24": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_11"]
        },
        "postdoc_bay_2": {
            "assets": {
                "desk_33": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "frame_3": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
                "desk_34": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_12"]
        },
        "lobby": {
            "assets": {
                "parcel": {
                    "accessible": True,
                    "affordance": ["pick", "drop"],
                    "state": "free",
                    "relation": "on"
                },
                "shelf_2": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {},
                },
            },
            "neighbor": ["corridor_14"]
        },
        "supplies_station": {
            "assets": {
                "cupboard_1": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "printer_paper": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "paper_towel": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
                "cupboard_2": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "vodka": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "orange_juice": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "biscuits": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "bottle_water_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "bottle_water_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "bottle_water_3": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "bottle_water_4": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "bottle_water_5": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
            },
            "neighbor": ["corridor_24"]
        },
        "printing_zone_1": {
            "assets": {},
            "neighbor": ["corridor_24"]
        },
        "ajays_office": {
            "assets": {},
            "neighbor": ["corridor_23"]
        },
        "chris_office": {
            "assets": {},
            "neighbor": ["corridor_22"]
        },
        "lauriannes_office": {
            "assets": {},
            "neighbor": ["corridor_22"]
        },
        "dimitys_office": {
            "assets": {
                "desk_3": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "K31X": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "buzzer": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        }
                    },
                },
                "cabinet_3": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "apple_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        }
                    },
                },
            },
            "neighbor": ["corridor_22"]
        },
        "meeting_room_3": {
            "assets": {
                "table_1": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "headphones": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "table_6": {
                    "accessible": False,
                    "affordance": [],
                    "state": "free",
                    "items": {},
                },
            },
            "neighbor": ["corridor_26"]
        },
        "agriculture_lab": {
            "assets": {
                "produce_container": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "kale_leaves_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_24"]
        },
        "kitchen": {
            "assets": {
                "cabine_1": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {},
                },
                "fridge": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "banana_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "cheese": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "tomato": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "J64M": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "chicken_kebab": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                        "noodles": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "on"
                        },
                        "salmon_bagel": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "salad": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "carrot": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "coffee_machine": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "off",
                    "items": {},
                },
                "kitchen_bench": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "kale_leaves_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "orange_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "bread": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "butter": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "chips": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
                "dishwasher": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "bowl": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "spoon": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                        "plate_2": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
                "drawer": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {},
                },
                "microvawe": {
                    "accessible": True,
                    "affordance": ["turnon", "turnoff"],
                    "state": "off",
                    "items": {},
                },
                "recycling_bin": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "milk_catron": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "in"
                        },
                        "orange_peel": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "in"
                        },
                        "apple_core": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "in"
                        },
                        "banana_peel": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
                "rubbish_bin": {
                    "accessible": True,
                    "affordance": ["open", "close"],
                    "state": "closed",
                    "items": {
                        "plastic_bottle": {
                            "accessible": False,
                            "affordance": [],
                            "state": "free",
                            "relation": "in"
                        },
                    },
                },
            },
            "neighbor": ["corridor_25"]
        },
        "cafeteria": {
            "assets": {
                "lunch_table": {
                    "accessible": True,
                    "affordance": [],
                    "state": "free",
                    "items": {
                        "banana_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "plate_1": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "fork": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                        "knife": {
                            "accessible": True,
                            "affordance": ["pick", "drop"],
                            "state": "free",
                            "relation": "on"
                        },
                    },
                },
            },
            "neighbor": ["corridor_25"]
        },
        "presentation_lounge": {
            "assets": {},
            "neighbor": ["corridor_26"]
        },
        "corridor_1": {
            "neighbor": ["peters_office", "tobis_office", "corridor_2"]
        },
        "corridor_2": {
            "neighbor": ["corridor_1", "meeting_room_1", "corridor_3", "corridor_5"]
        },
        "corridor_3": {
            "neighbor": ["corridor_2", "printing_zone_2", "robot_lounge_1", "corridor_9"]
        },
        "corridor_4": {
            "neighbor": ["postdoc_bay_1", "corridor_5"]
        },
        "corridor_5": {
            "neighbor": ["corridor_2", "corridor_4", "corridor_6", "corridor_7", "meeting_room_2", "nikos_office"]
        },
        "corridor_6": {
            "neighbor": ["phd_bay_1", "corridor_5"]
        },
        "corridor_7": {
            "neighbor": ["corridor_5", "michaels_office", "corridor_8", "corridor_16", "corridor_17"]
        },
        "corridor_8": {
            "neighbor": ["corridor_7", "mobile_robotics_lab", "corridor_13"]
        },
        "corridor_9": {
            "neighbor": ["corridor_3", "robot_lounge_2", "aarons_office", "corridor_10", "corridor_11", "corridor_13"]
        },
        "corridor_10": {
            "neighbor": ["filiopes_office", "luis_office", "wills_office", "corridor_9"]
        },
        "corridor_11": {
            "neighbor": ["phd_bay_3", "corridor_9"]
        },
        "corridor_12": {
            "neighbor": ["postdoc_bay_2", "corridor_13"]
        },
        "corridor_13": {
            "neighbor": ["jasons_office", "corridor_8", "corridor_9", "corridor_12", "corridor_14"]
        },
        "corridor_14": {
            "neighbor": ["lobby", "corridor_13", "corridor_23"]
        },
        "corridor_15": {
            "neighbor": ["manipulation_lab", "corridor_18", "corridor_23"]
        },
        "corridor_16": {
            "neighbor": ["corridor_7", "corridor_17", "corridor_18"]
        },
        "corridor_17": {
            "neighbor": ["phd_bay_2", "corridor_7", "corridor_16", "corridor_18"]
        },
        "corridor_18": {
            "neighbor": ["admin", "corridor_15", "corridor_16", "corridor_19"]
        },
        "corridor_19": {
            "neighbor": ["phd_bay_4", "corridor_18", "corridor_20"]
        },
        "corridor_20": {
            "neighbor": ["meeting_room_4", "corridor_19", "corridor_21"]
        },
        "corridor_21": {
            "neighbor": ["postdoc_bay_3", "corridor_20", "corridor_22"]
        },
        "corridor_22": {
            "neighbor": ["chriss_office", "lauriannes_office", "dimitys_office", "corridor_21", "corridor_23"]
        },
        "corridor_23": {
            "neighbor": ["ajarss_office", "corridor_14", "corridor_15", "corridor_22", "corridor_24"]
        },
        "corridor_24": {
            "neighbor": ["supplies_station", "printing_zone_1", "agriculture_lab", "corridor_23", "corridor_25"]
        },
        "corridor_25": {
            "neighbor": ["kitchen", "cafeteria", "corridor_24", "corridor_26"]
        },
        "corridor_26": {
            "neighbor": ["meeting_room_3", "presentation_lounge", "corridor_25"]
        },
    },
    "agent": {
        "position": "mobile_robotics_lab",
        "state": "hand-free"
    },
}


def load_scene_graph(scene: str):
    return copy.deepcopy(eval(scene.upper()))


def extract_accessible_items_from_sg(sg: dict):
    accessible_items = []
    for room_name, room_data in sg["rooms"].items():
        if "assets" in room_data:
            assets = room_data.get("assets", {})
            for asset_name, asset_data in assets.items():
                if asset_data.get("accessible", True):
                    accessible_items_asset = []
                    items = asset_data.get("items", {})
                    for item_name, item_data in items.items():
                        if item_data.get("accessible", True):
                            accessible_items_asset.append(item_name)
                    accessible_items.append(
                        {"asset": asset_name, "items": accessible_items_asset})
        else:
            items = room_data.get("items", {})
            for item_name, item_data in items.items():
                if item_data.get("accessible", True):
                    accessible_items.append(item_name)
    return accessible_items


def prune_sg_with_item(sg: dict, item_keep: list):
    pruned_sg = copy.deepcopy(sg)
    for room_name, room_data in pruned_sg["rooms"].items():
        pruned_items = {}
        if "assets" in room_data and any("asset" in elem for elem in item_keep):
            pruned_assets = {}
            assets = room_data.get("assets", {})
            for asset_name, asset_data in assets.items():
                if any(asset_name in elem["asset"] for elem in item_keep):
                    pruned_assets[asset_name] = {}
                    asset_items = asset_data.get("items", {})
                    for item_name, item_data in asset_items.items():
                        if any(item_name in elem["items"] for elem in item_keep):
                            pruned_assets[asset_name][item_name] = item_data
                room_data["assets"] = pruned_assets
        else:
            room_items = room_data.get("items", {})
            for item_name, item_data in room_items.items():
                if item_name in item_keep:
                    pruned_items[item_name] = item_data
            room_data["items"] = pruned_items

    return pruned_sg


def count_rooms(sg: dict):
    return len(sg["rooms"])


def count_items(sg: dict):
    count = 0
    for room_name, room_data in sg["rooms"].items():
        if "assets" in room_data:
            assets = room_data.get("assets", {})
            for asset_name, asset_data in assets.items():
                items = asset_data.get("items", {})
                count += len(items)
        else:
            items = room_data.get("items", {})
            count += len(items)
    return count


def collapsed_sg(sg: dict):
    cg = {}
    cg["name"] = sg["name"]
    cg["rooms"] = {}
    cg["agent"] = sg["agent"]
    for room_name, room_data in sg["rooms"].items():
        cg["rooms"][room_name] = {"items": {},
                                  "neighbor": room_data["neighbor"]}
    return cg


def update_sg(sg: dict, orig_sg: dict, command: str, room: str):
    if command == "expand":
        assert sg["rooms"][room]["items"] == {}, "Room is not empty!"
        sg["rooms"][room]["items"] = orig_sg["rooms"][room]["items"]
    elif command == "contract":
        # assert sg["rooms"][room]["items"] != {}, "Room is empty!"
        sg["rooms"][room]["items"] = {}
    else:
        raise Exception("Invalid command!")
    return sg
