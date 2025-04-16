(define (problem shelbiana_clean)
    (:domain clean)

    ; Begin objects
    (:objects
        robot - agent
        bathroom_1 bathroom_2 balcony bedroom_1 bedroom_2 bedroom_3 closet corridor_1 corridor_2 kitchen living_room lobby - room
        sink_1 sink_2 mop cola_can banana_peel rotting_apple rubbish_bin robot_hub - item
    )
    ; End objects

    ; Begin init
    (:init
        ; Connections
        (neighbor bathroom_1 lobby)
        (neighbor bathroom_2 corridor_2)
        (neighbor balcony living_room)
        (neighbor balcony bedroom_3)
        (neighbor closet corridor_2)
        (neighbor corridor_1 corridor_2)
        (neighbor corridor_1 kitchen)
        (neighbor corridor_1 living_room)
        (neighbor corridor_2 bathroom_2)
        (neighbor corridor_2 bedroom_2)
        (neighbor corridor_2 bedroom_3)
        (neighbor corridor_2 closet)
        (neighbor corridor_2 corridor_1)
        (neighbor living_room balcony)
        (neighbor living_room corridor_1)
        (neighbor living_room lobby)
        (neighbor bedroom_1 lobby)
        (neighbor bedroom_2 corridor_2)
        (neighbor bedroom_3 balcony)
        (neighbor bedroom_3 corridor_2)
        (neighbor lobby bathroom_1)
        (neighbor lobby bedroom_1)
        (neighbor lobby living_room)
        (neighbor kitchen corridor_1)

        ; Positions
        (agent_at robot bedroom_3)
        (item_at sink_1 bathroom_1)
        (item_at sink_2 bathroom_2)
        (item_at mop bathroom_1)
        (item_at cola_can living_room)
        (item_at banana_peel living_room)
        (item_at rotting_apple bedroom_2)
        (item_at rubbish_bin kitchen)
        (item_at robot_hub bedroom_3)

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