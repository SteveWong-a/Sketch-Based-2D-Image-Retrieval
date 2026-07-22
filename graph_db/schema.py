class Neo4jSchemaManager:
    """
    Manages the mock schema for the Graph Database (Neo4j).
    Stores 2D image assets as nodes containing metadata and an HNSW vector index.
    """
    
    @staticmethod
    def get_index_creation_queries() -> list[str]:
        """
        Returns the Cypher queries needed to set up the HNSW vector index for the 512-D CLIP embeddings.
        """
        queries = [
            # Drop index if it exists
            "DROP INDEX clip_image_index IF EXISTS",
            
            # Create a vector index on the 'clip_embedding' property of 'ImageAsset' nodes
            """
            CREATE VECTOR INDEX clip_image_index IF NOT EXISTS
            FOR (i:ImageAsset)
            ON (i.clip_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 512,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        ]
        return queries

    @staticmethod
    def get_mock_data_queries() -> list[str]:
        """
        Returns Cypher queries to populate mock image nodes.
        In a real scenario, embeddings would be actual 512-D vectors, here we mock them.
        """
        queries = [
            """
            CREATE (i1:ImageAsset {
                id: 'img_001',
                title: 'Sunny Mountain',
                author: 'Alice',
                style: 'photograph',
                tags: ['nature', 'mountain', 'landscape'],
                quality: 0.95
            })
            """,
            """
            CREATE (i2:ImageAsset {
                id: 'img_002',
                title: 'City Skyline Sketch',
                author: 'Bob',
                style: 'sketch',
                tags: ['city', 'urban', 'buildings'],
                quality: 0.80
            })
            """
        ]
        return queries
