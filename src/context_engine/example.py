from context_engine.code_index import CodeEntityStore, OnnxTextEmbedder
from models.embedding_models import CodeItemIn, EntityType, SearchQuery
from context_engine.config import (
    EMBED_ONNX,
TOKENIZER,
INDEX_DB,
)




def main() -> None:
    embedder = OnnxTextEmbedder(
        model_path=EMBED_ONNX,
        tokenizer_path=TOKENIZER,
        max_length=512,
        batch_size=32,
    )

    store = CodeEntityStore(
        db_path=INDEX_DB,
        embedder=embedder,
        preload=True,
    )

    # ------------------------------------------------------------------
    # Insert examples
    # ------------------------------------------------------------------
    store.add_items(
        [
            CodeItemIn(
                entity_type=EntityType.FUNCTION,
                source_id=10,
                name="get_user_id",
                extra_text=None,
            ),
            CodeItemIn(
                entity_type=EntityType.FUNCTION,
                source_id=3,
                name="auth_user",
                extra_text="authenticate a user login",
            ),

        ]
    )

    # # ------------------------------------------------------------------
    # # Search by meaning, not exact name
    # # ------------------------------------------------------------------
    # results = store.search(
    #     "make a new user",
    #     limit=10,
    # )

    # print("Search results:")
    # for result in results:
    #     print(
    #         result.entity_type.value,
    #         result.source_id,
    #         result.name,
    #         result.score,
    #     )

    # ------------------------------------------------------------------
    # Search only functions
    # ------------------------------------------------------------------
    function_results = store.search(
        SearchQuery(
            query="get user ID",
            entity_types=[EntityType.FUNCTION],
            limit=5,
            min_score=-1.0,
        )
    )

    print("\nFunction-only results:")
    for result in function_results:
        print(
            result.entity_type.value,
            result.source_id,
            result.name,
            result.score,
        )

    function_results2 = store.search(
        SearchQuery(
            query="password and username correct",
            entity_types=[EntityType.FUNCTION],
            limit=5,
            min_score=-1.0,
        )
    )

    print("\nFunction-only results:")
    for result in function_results2:
        print(
            result.entity_type.value,
            result.source_id,
            result.name,
            result.score,
        )


    

    # ------------------------------------------------------------------
    # Use result to query your original database
    # ------------------------------------------------------------------
    # Example:
    #
    # for result in results:
    #     if result.entity_type == EntityType.FUNCTION:
    #         row = original_db.execute(
    #             "SELECT * FROM functions WHERE id = ?",
    #             (result.source_id,),
    #         ).fetchone()
    #
    #     elif result.entity_type == EntityType.CLASS:
    #         row = original_db.execute(
    #             "SELECT * FROM classes WHERE id = ?",
    #             (result.source_id,),
    #         ).fetchone()
    #
    #     elif result.entity_type == EntityType.METHOD:
    #         row = original_db.execute(
    #             "SELECT * FROM methods WHERE id = ?",
    #             (result.source_id,),
    #         ).fetchone()

    store.close()


if __name__ == "__main__":
    main()