(define (problem parole_pc)
    (:domain pc)

    ; Begin objects
    (:objects
        robot - agent
        bathroom bedroom corridor_1 corridor_2 dining_room kitchen living_room - room
        mainboard cpu ram ssd gpu psu - item
        my_pc - pc
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
        (item_at psu bathroom)
        (item_at cpu bedroom)
        (item_at ssd corridor_1)
        (item_at gpu corridor_2)
        (item_at ram dining_room)
        (item_at mainboard kitchen)

        ; Attributes
        (item_is_mainboard mainboard)
        (item_accessible mainboard)
        (item_pickable mainboard)
        (item_is_cpu cpu)
        (item_accessible cpu)
        (item_pickable cpu)
        (item_is_ram ram)
        (item_accessible ram)
        (item_pickable ram)
        (item_is_ssd ssd)
        (item_accessible ssd)
        (item_pickable ssd)
        (item_is_gpu gpu)
        (item_accessible gpu)
        (item_pickable gpu)
        (item_is_psu psu)
        (item_accessible psu)
        (item_pickable psu)
    )
    ; End init

    ; Begin goal
    (:goal
        (and
            (item_at psu living_room)
            (item_at gpu living_room)
            (item_at ssd living_room)
            (item_at ram living_room)
            (item_at cpu living_room)
            (item_at mainboard living_room)
            (pc_assembled my_pc)
        )
    )
    ; End goal
)