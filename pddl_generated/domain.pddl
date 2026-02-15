(define (domain office)
    (:requirements :strips :typing :adl)
    (:types agent room item)

    (:predicates
        (neighbor ?r1 - room ?r2 - room)
        (agent_at ?a - agent ?r - room)
        (item_at ?i - item ?r - room)
        (agent_loaded ?a - agent)
        (agent_has_item ?a - agent ?i - item)
        (item_dirty ?i - item)
        (item_clean ?i - item)
        (item_is_sink ?i - item)
        
        ;; --- 汎用アフォーダンス述語 ---
        (can_graspable ?i - item)   ;; 掴むことができるか
        (can_liftable ?i - item)    ;; 持ち上げられるか（重すぎないか）
        (can_washable ?i - item)    ;; 洗浄可能か
        (can_accessible ?i - item)  ;; アクセス可能か（以前の item_accessible）
    )

    ;; 移動アクション
    (:action goto
        :parameters (?a - agent ?r1 - room ?r2 - room)
        :precondition (and (agent_at ?a ?r1) (neighbor ?r1 ?r2))
        :effect (and (not (agent_at ?a ?r1)) (agent_at ?a ?r2))
    )

    ;; 拾い上げアクション
    (:action pick
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and 
            (agent_at ?a ?r) 
            (item_at ?i ?r) 
            (can_accessible ?i)   ;; 障害物がない
            (can_graspable ?i)    ;; アフォーダンス：掴める形状か
            (can_liftable ?i)     ;; アフォーダンス：持ち上がる重さか
            (not (agent_loaded ?a))
        )
        :effect (and 
            (not (item_at ?i ?r)) 
            (agent_loaded ?a) 
            (agent_has_item ?a ?i)
        )
    )

    ;; 配置アクション
    (:action drop
        :parameters (?a - agent ?i - item ?r - room)
        :precondition (and (agent_at ?a ?r) (agent_has_item ?a ?i))
        :effect (and 
            (item_at ?i ?r) 
            (not (agent_loaded ?a)) 
            (not (agent_has_item ?a ?i))
        )
    )

    ;; 洗浄アクション
    (:action wash
        :parameters (?a - agent ?i - item ?s - item ?r - room)
        :precondition (and 
            (agent_at ?a ?r) 
            (item_at ?s ?r) 
            (item_is_sink ?s) 
            (agent_has_item ?a ?i) 
            (item_dirty ?i)
            (can_washable ?i)     ;; アフォーダンス：水で洗っても壊れないか
        )
        :effect (and 
            (not (item_dirty ?i)) 
            (item_clean ?i)
        )
    )
)