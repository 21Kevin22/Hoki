(define (problem allensville_clean)
    (:domain clean)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 bedroom_1 bedroom_2 corridor_1 corridor_2 corridor_3 dining_room kitchen living_room lobby - room
        sink_1 sink_2 mop cola_can banana_peel rotting_apple rubbish_bin robot_hub - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom_1 corridor_2)
        (neighbor bathroom_2 corridor_3)
        (neighbor bedroom_1 corridor_2)
        (neighbor bedroom_2 corridor_3)
        (neighbor corridor_1 lobby)
        (neighbor corridor_1 corridor_3)
        (neighbor corridor_2 bathroom_1)
        (neighbor corridor_2 bedroom_1)
        (neighbor corridor_2 corridor_3)
        (neighbor corridor_3 corridor_1)
        (neighbor corridor_3 corridor_2)
        (neighbor corridor_3 bathroom_2)
        (neighbor corridor_3 bedroom_2)
        (neighbor corridor_3 kitchen)
        (neighbor corridor_3 living_room)
        (neighbor dining_room kitchen)
        (neighbor dining_room living_room)
        (neighbor kitchen corridor_3)
        (neighbor kitchen dining_room)
        (neighbor living_room corridor_3)
        (neighbor living_room dining_room)
        (neighbor lobby corridor_1)

        ; Positions
        (agent_at robot living_room)
        (item_at sink_1 bathroom_1)
        (item_at sink_2 bathroom_2)
        (item_at mop bathroom_1)
        (item_at cola_can dining_room)
        (item_at banana_peel lobby)
        (item_at rotting_apple bedroom_2)
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