use rustc_hash::FxHashMap;

use super::kernels::Kernel;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Value types. A `Bool` is stored as a 0/1 column.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Num,
    Bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// `f64` bits, so the node can be hashed.
    Const(u64),
    Arg(u16),
    Op {
        kernel: Kernel,
        args: Vec<NodeId>,
    },
    /// `kernel`'s partial derivative with respect to argument `idx`, evaluated at `args`.
    Partial {
        kernel: Kernel,
        idx: u8,
        args: Vec<NodeId>,
    },
}

impl Node {
    pub fn args(&self) -> &[NodeId] {
        match self {
            Node::Const(_) | Node::Arg(_) => &[],
            Node::Op { args, .. } | Node::Partial { args, .. } => args,
        }
    }
}

/// A hash-consed DAG: structurally equal nodes share one id, and every node's arguments have
/// smaller ids than the node itself.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    index: FxHashMap<Node, NodeId>,
}

impl Graph {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    pub fn intern(&mut self, node: Node) -> NodeId {
        if let Some(&id) = self.index.get(&node) {
            return id;
        }
        let id = NodeId(u32::try_from(self.nodes.len()).expect("IR graph exceeds u32 nodes"));
        self.nodes.push(node.clone());
        self.index.insert(node, id);
        id
    }

    pub fn constant(&mut self, value: f64) -> NodeId {
        self.intern(Node::Const(value.to_bits()))
    }

    pub fn arg(&mut self, i: u16) -> NodeId {
        self.intern(Node::Arg(i))
    }

    pub fn op(&mut self, kernel: Kernel, args: Vec<NodeId>) -> NodeId {
        debug_assert_eq!(args.len(), usize::from(kernel.arity()));
        self.intern(Node::Op { kernel, args })
    }

    pub fn partial(&mut self, kernel: Kernel, idx: u8, args: Vec<NodeId>) -> NodeId {
        debug_assert_eq!(args.len(), usize::from(kernel.arity()));
        self.intern(Node::Partial { kernel, idx, args })
    }

    pub fn const_value(&self, id: NodeId) -> Option<f64> {
        match self.node(id) {
            Node::Const(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Marks the nodes reachable from `roots`, indexed by node id.
    pub fn reachable(&self, roots: &[NodeId]) -> Vec<bool> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !std::mem::replace(&mut reachable[id.index()], true) {
                stack.extend_from_slice(self.node(id).args());
            }
        }
        reachable
    }

    /// Copies the nodes reachable from `roots` into a new graph, keeping their relative order.
    pub fn extract(&self, roots: &[NodeId]) -> (Graph, Vec<NodeId>) {
        let reachable = self.reachable(roots);
        let mut out = Graph::default();
        let mut map = vec![NodeId(u32::MAX); self.nodes.len()];
        for (i, node) in self.nodes.iter().enumerate() {
            if reachable[i] {
                map[i] = out.intern(remap(node, &map));
            }
        }
        let roots = roots.iter().map(|r| map[r.index()]).collect();
        (out, roots)
    }

    /// Copies `callee`'s nodes up to `root` into `self`, substituting `args[i]` for `Arg(i)`.
    /// Returns the id of the copied root.
    pub fn inline(&mut self, callee: &Graph, root: NodeId, args: &[NodeId]) -> NodeId {
        let mut map = Vec::with_capacity(root.index() + 1);
        for node in &callee.nodes[..=root.index()] {
            let id = match node {
                Node::Arg(i) => args[usize::from(*i)],
                node => self.intern(remap(node, &map)),
            };
            map.push(id);
        }
        map[root.index()]
    }
}

fn remap(node: &Node, map: &[NodeId]) -> Node {
    let remap_args = |args: &[NodeId]| -> Vec<NodeId> { args.iter().map(|a| map[a.index()]).collect() };
    match node {
        Node::Const(_) | Node::Arg(_) => node.clone(),
        Node::Op { kernel, args } => Node::Op {
            kernel: *kernel,
            args: remap_args(args),
        },
        Node::Partial { kernel, idx, args } => Node::Partial {
            kernel: *kernel,
            idx: *idx,
            args: remap_args(args),
        },
    }
}
