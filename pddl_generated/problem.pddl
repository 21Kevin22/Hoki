(define (problem office_task)
    (:domain office)
    (:objects
        cup - item
        agent1 - agent
        office kitchen - room
        sink1 - item
    )
    (:init 
        ;; --- 基本状態 ---
        (item_at cup office)
        (agent_at agent1 office)
        (neighbor office kitchen)
        (neighbor kitchen office)
        (item_at sink1 kitchen)
        (item_is_sink sink1)
        
        ;; --- 汎用アフォーダンス属性 (Affordances) ---
        ;; JSONから抽出された属性を can_ プレフィックスで定義
        (can_graspable cup)    ;; 掴むことができる
        (can_liftable cup)     ;; 持ち上げることができる
        (can_washable cup)     ;; 洗うことができる
        (can_fillable cup)     ;; 水を入れることができる
        
        (can_accessible cup)   ;; アクセス可能（障害物がない）
        (can_accessible sink1) ;; シンクが利用可能
        
        ;; --- 状態（アフォーダンスではない動的な状態） ---
        (item_dirty cup) 
        ;; agent_loaded は記述なし = 手が空いている
    )
    (:goal 
        (and (item_clean cup))
    )
)