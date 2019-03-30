gdca = (function() {
    'use strict';

    var gdca = {
        cf: null,
    };

    gdca.pareto = (function() {
        var pareto = {
            dom: [],
            topI: [],
        };
        return {
            getDomain: function() {
                return pareto.dom; // could pluck the key, but the work was already done
            },
            domain: function(grp) {
                var all = gdca.cf.groupAll();

                var sum = 0;
                // .all() is faster than .top(Infinity)
                pareto.topI = grp.top(Infinity);
                pareto.dom = [];
                pareto.topI.forEach(function(d) {
                    pareto.dom[pareto.dom.length] = d.key;
                    sum = sum + d.value;
                    d.sum = sum;
                    d.pp = 100 * sum / all.value();
                });
            },
            createTempOrderingGroupReversed: function() {
                var grp = {
                    all: function() {
                        var g = [];
                        pareto.topI.forEach(function(d, i) {
                            g.push({ key: d.key, value: d.pp });
                        });
                        return g;
                    },
                };
                return grp;
            },
            createTempOrderingGroup: function() {
                var grp = {
                    all: function() {
                        var g = [];
                        pareto.topI.forEach(function(d, i) {
                            g.push({ key: d.key, value: d.value });
                        });
                        return g;
                    },
                };
                return grp;
            },
        };
        return gdca.pareto;
    })();

    return gdca;
})();
