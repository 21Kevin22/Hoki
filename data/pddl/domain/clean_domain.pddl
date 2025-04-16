;Header and description
(define (domain clean)

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item
    )
    ; End types

    ; Begin predicates
    (:predicates
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (item_pickable ?i - item)
        (item_accessible ?i - item)

        (neighbor ?r1 - room ?r2 - room)

        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)

        (item_is_mop ?i - item)
        (item_is_sink ?i - item)
        (item_is_rubbish_bin ?i - item)
        (item_is_robot_hub ?i - item)
        (item_disposed ?i - item)
        (floor_clean ?r - room)
        (mop_clean ?i - item)
        (battery_full ?a - agent)
    )
    ; End predicates

    ; Begin actions
    (:action goto
        :parameters (?a - agent ?r1 - room ?r2 - room)
        :precondition (and
            (agent_at ?a ?r1)
            (neighbor ?r1 ?r2)
        )
        :effect (and
            (not(agent_at ?a ?r1))
            (agent_at ?a ?r2)
        )
    )

    (:action pick
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_accessible ?i)
            (item_pickable ?i)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
        )
        :effect (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
        )
    )

    (:action drop
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (item_accessible ?i)
            (item_pickable ?i)
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
        )
        :effect (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i))
        )
    )

    (:action dispose
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i2 ?r)
            (item_accessible ?i1)
            ; (item_accessible ?i2)
            (item_pickable ?i1)
            (item_is_rubbish_bin ?i2)
            (agent_loaded ?a)
            (agent_has_item ?a ?i1)
            (not(item_disposed ?i1))
        )
        :effect (and
            (item_disposed ?i1)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i1))
            (not(battery_full ?a))
        )
    )

    (:action mop_floor
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_accessible ?i)
            (item_pickable ?i)
            (item_is_mop ?i)
            (agent_loaded ?a)
            (agent_has_item ?a ?i)
            (not(floor_clean ?r))
            (mop_clean ?i)
        )
        :effect (and
            (floor_clean ?r)
            (not(mop_clean ?i))
            (not(battery_full ?a))
        )
    )

    (:action clean_mop
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i2 ?r)
            (item_accessible ?i1)
            ; (item_accessible ?i2)
            (item_pickable ?i1)
            (item_is_mop ?i1)
            (agent_loaded ?a)
            (agent_has_item ?a ?i1)
            (item_is_sink ?i2)
            (not(mop_clean ?i1))
        )
        :effect (and
            (mop_clean ?i1)
            (item_at ?i1 ?r)
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i1))
            (not(battery_full ?a))
        )
    )

    (:action charge
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_accessible ?i)
            (item_is_robot_hub ?i)
            (not(battery_full ?a))
            (not(agent_loaded ?a))
        )
        :effect (and
            (battery_full ?a)
        )
    )
    ; End actions
)