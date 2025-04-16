;Header and description
(define (domain home)

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
        (item_on ?i - item)
        (item_pickable ?i - item)
        (item_turnable ?i - item)
        (item_accessible ?i - item)

        (neighbor ?r1 ?r2 - room)

        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
    )
    ; End predicates

    ; Begin actions
    (:action goto
        :parameters (?a - agent ?r1 ?r2 - room)
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

    (:action turnon
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_accessible ?i)
            (item_turnable ?i)
            (not(item_on ?i))
        )
        :effect (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_on ?i)
        )
    )

    (:action turnoff
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_accessible ?i)
            (item_turnable ?i)
            (item_on ?i)
        )
        :effect (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (not(item_on ?i))
        )
    )
    ; End actions

)