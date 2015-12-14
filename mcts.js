/*
Author: Youngjun Kim, youngjun@stanford.edu
Date: 12/13/2015
*/


var datafile = document.getElementById("mctsVisualizer").getAttribute("datafile")
var collapse_flag = document.getElementById("mctsVisualizer").getAttribute("collapse") || false

var width = 1200,
    height = 1200;

var tree = d3.layout.tree()
    .size([width, height])
    .children(function (d) { return d.states || d.actions; });

var diagonal = d3.svg.diagonal();

var svg = d3.select("body").append("svg")
    .attr("width", width + 200)
    .attr("height", height + 40)
    .append("g")
    .attr("transform", "translate(100,20)");

var root,
    i = 0,
    duration = 750;

d3.json(datafile, function(error, json) {
    root = json
    root.x0 = width / 2;
    root.y0 = 0;

    if (collapse_flag)
        root.forEach(collapse);

    update(root);
});


function collapse(d) {
    if (d.states) {
        d._states = d.states;
        d._states.forEach(collapse);
        d.states = null;
    } else if (d.actions) {
        d._actions = d.actions;
        d._actions.forEach(collapse);
        d.actions = null;
    }
}


function click(d) {
    if (d.states) {
        d._states = d.states;
        d.states = null;
    } else if (d._states) {
        d.states = d._states;
        d._states = null;
    } else if (d.actions && d.actions.length != 0) {
        d._actions = d.actions;
        d.actions = null;
    } else if (d._actions) {
        d.actions = d._actions;
        d._actions = null;
    }

    update(d);
}


function mouseover(d) {
    d3.select(this).append("text")
        .attr("class", "hover")
        .attr("text-anchor", "middle")
        .attr('transform', function(d) { return 'translate(0, -8)'; })
        .text(function (d) {
            if (d.state)
                return d.state + ", " + d.n + ", " + d.N
            else if (d.action)
                return d.action + ", " + d.n + ", " + d.r.toPrecision(4) + ", " + d.N + ", " + d.Q.toPrecision(4)
        });
}


function mouseout(d) {
    d3.select(this).select("text.hover").remove();
}


function update(source) {
    var nodes = tree.nodes(root).reverse(),
        links = tree.links(nodes);

    //console.log(nodes)

    nodes.forEach(function(d) { d.y = d.depth * 50; });

    var node = svg.selectAll("g.node")
        .data(nodes, function(d) { return d.id || (d.id = ++i); });


    var nodeEnter = node.enter().append("g")
        .attr("class", "node")
        .attr("transform", function(d) { return "translate(" + source.x0 + "," + source.y0 + ")"; })
        .on("click", click)
        .on("mouseover", mouseover)
        .on("mouseout", mouseout);

    nodeEnter.append("path")
        .attr("d", d3.svg.symbol()
            .type(function(d) {
                if (d.state)
                    return "circle";
                else if (d.action)
                    return "square";
        }))
        .attr("style", coloring_mf_node);


    var nodeUpdate = node.transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

    nodeUpdate.select("path")
        .attr("d", d3.svg.symbol()
            .type(function(d) {
                if (d.state)
                    return "circle";
                else if (d.action)
                    return "square";
        }))
        .attr("style", coloring_mf_node);


    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + source.x + "," + source.y + ")"; })
        .remove();

    nodeExit.select("path")
        .attr("d", d3.svg.symbol()
            .size(1e-6)
            .type(function(d) {
                if (d.state)
                    return "circle";
                else if (d.action)
                    return "square";
        }));


    var link = svg.selectAll("path.link")
        .data(links, function(d) { return d.target.id; });

    link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", function(d) {
            var o = {x: source.x0, y: source.y0};
            return diagonal({source: o, target: o});
        });

    link.transition()
        .duration(duration)
        .attr("d", diagonal);

    link.exit().transition()
        .duration(duration)
        .attr("d", function(d) {
            var o = {x: source.x, y: source.y};
            return diagonal({source: o, target: o});
        })
        .remove();


    nodes.forEach(function(d) {
        d.x0 = d.x;
        d.y0 = d.y;
    });
}


function coloring_mf_node(d) {
    var children;

    if (d.state) {
        style = "fill: purple"
        children = d.actions
    } else if (d.action) {
        if (d.action == "up")
            style = "stroke: green"
        else if (d.action == "right")
            style = "stroke: blue"
        children = d.states
    }

    if (children != null)
        style += "; stroke-opacity: 0.5; fill-opacity: 0.5"
    else
        style += "; stroke-opacity: 1.0; fill-opacity: 1.0"

    return style
}


