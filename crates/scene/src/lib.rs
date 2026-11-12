//! Ganbate's authoring model, shared by the editor, the cooker and the run-time loaders: scene and
//! prefab documents (`scene`), the component schemas that describe what an entity can carry
//! (`schema`), undoable edits on any document (`doc`), and sibling order keys (`order`).
//! Everything here is portable: no files, threads or GPU.
pub mod doc;
pub mod order;
pub mod scene;
pub mod schema;
