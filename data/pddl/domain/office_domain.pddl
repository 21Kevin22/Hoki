(define (domain office) ; 

    (:requirements :strips :typing :adl)

    ; Begin types
    (:types
        agent room item
    )
    ; End types

    ; Begin predicates
    (:predicates
        (neighbor ?r1 - room ?r2 - room)
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (item_pickable ?i - item)
        (item_loadable ?i - item)
        (item_accessible ?i - item)
        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
        (item_in ?i1 - item ?i2 - item)
        (item_empty ?i - item)
        (item_dirty ?i - item)
        (item_clean ?i - item)
        (item_closed ?i - item)
        (item_open ?i - item)

        (item_is_sink ?i - item) 
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

    (:action pick_loadable
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_accessible ?i)
            (item_loadable ?i)
            (item_empty ?i)
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

    (:action drop_loadable
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (not(item_at ?i ?r))
            (item_accessible ?i)
            (item_loadable ?i)
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

    (:action load
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_loadable ?i1)
            (item_empty ?i1)
            (agent_loaded ?a)
            (agent_has_item ?a ?i2)
            (not(item_in ?i2 ?i1))
        )
        :effect (and
            (item_in ?i2 ?i1)
            (not(item_at ?i2 ?r))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i2))
            (not(item_empty ?i1))
        )
    )

    (:action unload
        :parameters (?a - agent ?i1 - item ?i2 - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i1 ?r)
            (item_loadable ?i1)
            (not(item_empty ?i1))
            (not(agent_loaded ?a))
            (not(agent_has_item ?a ?i2))
            (item_in ?i2 ?i1)
        )
        :effect (and
            (not(item_in ?i2 ?i1))
            (item_at ?i2 ?r)
            (item_empty ?i1)
        )
    )

    ; --- 修正箇所: wash アクション ---
    (:action wash
        ; パラメータに ?s (シンク) を追加
        :parameters (?a - agent ?i - item ?s - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (agent_has_item ?a ?i)
            (item_dirty ?i)
            (item_accessible ?i)
            
            ; シンクが同じ部屋にあること
            (item_at ?s ?r)
            ; それがシンクであること
            (item_is_sink ?s)
        )
        :effect (and
            (not(item_dirty ?i))
            (item_clean ?i)
        )
    )

    (:action open
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and
            (agent_at ?a ?r)
            (item_at ?i ?r)
            (item_closed ?i)
            (item_accessible ?i)
        )
        :effect (and
            (not(item_closed ?i))
            (item_open ?i)
        )
    )
    ; End actions
)