(define (problem parole_clean)
    (:domain clean)

    ; Begin objects
    (:objects
        robot - agent
        bathroom bedroom corridor_1 corridor_2 dining_room kitchen living_room - room
        sink_1 sink_2 mop cola_can banana_peel rotting_apple rubbish_bin robot_hub - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom corridor_2)
        (neighbor bedroom corridor_2)
        (neighbor corridor_1 corridor_2)
        (neighbor corridor_1 dining_room)
        (neighbor corridor_1 kitchen)
        (neighbor corridor_2 bathroom)
        (neighbor corridor_2 bedroom)
        (neighbor corridor_2 corridor_1)
        (neighbor dining_room corridor_1)
        (neighbor dining_room living_room)
        (neighbor kitchen corridor_1)
        (neighbor living_room dining_room)

        ; Positions
        (agent_at robot living_room)
        (item_at sink_1 bathroom)
        (item_at sink_2 bathroom)
        (item_at mop corridor_1)
        (item_at cola_can dining_room)
        (item_at banana_peel corridor_2)
        (item_at rotting_apple bedroom)
        (item_at rubbish_bin kitchen)
        (item_at robot_hub living_room)
        
        ; Attributes
        (item_is_mop mop)
        (item_is_sink sink_1)
        (item_is_sink sink_2)
        (item_is_rubbish_bin rubbish_bin)
        (item_is_robot_hub robot_hub)
        (mop_clean mop)
        (item_accessible mop)
        (item_pickable mop)
        (item_accessible sink_1)
        (item_accessible sink_2)
        (item_accessible rubbish_bin)
        (item_accessible robot_hub)
        (item_accessible cola_can)
        (item_pickable cola_can)
        (item_accessible banana_peel)
        (item_pickable banana_peel)
        (item_accessible rotting_apple)
        (item_pickable rotting_apple)
        (battery_full robot)
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (item_disposed cola_can)
            (item_disposed banana_peel)
            (item_disposed rotting_apple)
            (floor_clean living_room)
            (floor_clean kitchen)
            (mop_clean mop)
            (battery_full robot)
        )
    )
    ; End goal
)