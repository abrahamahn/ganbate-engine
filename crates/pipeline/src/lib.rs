//! Ganbate's content pipeline: a project folder and its asset database (`project`), importers that
//! turn source files into documents without losing what they are made of, content-addressed
//! Derived Data Cache (`ddc`), and the cooker that turns documents into what the runtime loads.
pub mod cook;
pub mod ddc;
pub mod project;
