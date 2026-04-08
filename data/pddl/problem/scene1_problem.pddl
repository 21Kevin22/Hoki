(define (problem scene1)
    (:domain home)
    (:objects 
        panda - agent
        bottle_0 bottle_1 bottle_2 - item
        table basket - room
    )
    (:init
        (agent_at panda table)
        (neighbor table basket)
        (neighbor basket table)
        (item_at bottle_0 table)
        (item_pickable bottle_0)
        (item_accessible bottle_0)
        (item_at bottle_1 table)
        (item_pickable bottle_1)
        (item_accessible bottle_1)
        (item_at bottle_2 table)
        (item_pickable bottle_2)
        (item_accessible bottle_2)
    )
    (:goal (and 
        (item_at bottle_0 basket)
        (item_at bottle_1 basket)
        (item_at bottle_2 basket)
    ))
)