# Team-2

# API Documentation
- `../route.py` documentation
- This document outlines the usage of the `/summarize` and `/resummarize` API endpoints.

## Endpoints

### 1. `/summarize`

#### **Method**: `POST`

#### **Description**
This endpoint performs text summarization by segmenting, clustering, and summarizing the input text or a `.txt` file.

#### **Request Format**
The request must be in JSON format and support either direct text input or file upload (not both).

#### **Request Parameters**
| Parameter       | Type     | Required | Description                            |
|-----------------|----------|----------|----------------------------------------|
| `select_model`  | `string` | Yes      | The model to use for summarization.    |
| `text`          | `string` | Optional | Direct input text for summarization.   |
| `file`          | `file`   | Optional | A `.txt` file containing text to summarize. |

**Note**: Either `text` or `file` must be provided, but not both.

#### **Example Request**
```json
{
    "select_model": "default_model",
    "text": "This is a sample text for summarization."
}
```

#### **Example Response**
```json
{
    "batch_summaries": ["Summary 1", "Summary 2"],
    "batch_importances": [[0.8, 0.7, ...], [0.2, 0.1, ...],
    "evaluation_results": {
        "rouge1": 85.2,
        "rouge2": 75.4,
        "rougeL": 80.1,
        "bert_score": 90.5
    },
    "segments": ["Segment 1", "Segment 2"],
    "concat_indices": [[0, 1, ...], [3, 4, ...]],         # segment index by theme
    "visualize_image": "<binary_data_encoded_as_ISO-8859-1_string>"
}
```

#### **Errors**
| Status Code | Message                                       |
|-------------|-----------------------------------------------|
| 400         | Request must be in JSON format.              |
| 400         | Either text or file must be provided, but not both. |
| 400         | Text content is empty.                       |
| 500         | Internal server error.                       |

---

### 2. `/resummarize`

#### **Method**: `POST`

#### **Description**
This endpoint filters segments of the `full_text` based on semantic similarity with the `target_text` and generates a summary of the filtered segments.

#### **Request Format**
The request must be in JSON format.

#### **Request Parameters**
| Parameter      | Type     | Required | Description                            |
|----------------|----------|----------|----------------------------------------|
| `full_text`    | `string` | Yes      | The complete text to be segmented and filtered. |
| `target_text`  | `string` | Yes      | The text to filter segments by semantic similarity. |

#### **Example Request**
```json
{
    "full_text": "This is the full text containing multiple segments.",
    "target_text": "target segment for filtering"
}
```

#### **Example Response**
```json
{
    "summary": "This is the generated summary from filtered text."
}
```

#### **Errors**
| Status Code | Message                                       |
|-------------|-----------------------------------------------|
| 400         | Request must be in JSON format.              |
| 400         | Both full_text and target_text are required. |
| 500         | Internal server error.                       |

---

## Additional Notes

1. **File Handling**
   - Only `.txt` files are allowed for the `file` parameter in the `/summarize` endpoint.

2. **Response for Visualized Image**
   - The `visualize_image` field in the `/summarize` response contains the binary data of the PNG file encoded as an ISO-8859-1 string.
   - To render the image on the client side, decode it back to binary.

3. **Error Responses**
   - Ensure proper request format and parameters to avoid common errors.

4. **Performance**
   - Large inputs may result in longer processing times; handle timeouts accordingly.

