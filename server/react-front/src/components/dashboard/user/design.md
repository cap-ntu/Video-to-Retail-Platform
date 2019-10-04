# Structure design
A form is consist of components and a host. The host is in charge of dispatching validation check, display components with
states and conduct submission.

# Host
Host is a top layer component:
```javascript
import React from "react";

class TopLayer extends React.PureComponent {
    state = {
        component: "",
        // validation store
        validation: {
            component: {error: false},
        }
    };

    // collect validation result and store in `this.state.validation`
    collectValidation() {
        // some code
    }

    // collect text field value
    handleChange = name => event => {
        this.setState({[name]: event.target.value});
    };

    // submit handler
    handleSubmit = () => {
        // some code
    };

    render() {
        const {component} = this.state;

        return (
            <div className="root">
                <Component value={component}/>
                <button onClick={this.handleSubmit}>
                    Done
                </button>
            </div>
        )
    }
}
```


# Component
# Control Flow
```
 +------------------+
 |       Host       |
 +------------------+
     |           ^
     |           | 3. updateValidation
     | 1. render |     ----------
     |           |     |        |
     v           |     v        | 2. validate
 +----------------------+       |
 |       Component      |--------
 +----------------------+  (blur)
```
## Representation
Components as the children of the Host, is rendered as normal React Component.
## Logic
The validation process happens in these two scenarios:
1. when submit form is clicked
2. when user finish one text field and leave (onBlur event)

Therefore, Host need to access to each validate function of component. Meanwhile, components need to submit validation
result back to Host.

# Component Example
## Group
### Fine transition of group
Group or a list of some component is use `Array.prototype.map` function to dynamically render
```javascript
import React from "react";

class Group extends React.PureComponent {
    render() {
        return (
            <div> 
            {
                [1, 2, 3, 4, 5].map(i =>
                    <div key={i}>
                        {i}
                    </div>
                )
            }
            </div>
        )
    }
}
```
Problem arises that when we use dynamically determined mapped list
```javascript
import React from "react";

class Group extends React.PureComponent {
    state = {
        list: [1, 2, 3, 4, 5]
    };
    
    handleAdd = () => {
        const {list} = this.state;
        // do not directly modify state, even it is a reference. Create a copy and dispatch the modified copy 
        this.setState({list: [...list, list[list.length - 1] + 1]});    // this will have race condition
    };
    
    handleDrop = () => {
            const {list} = this.state;
            this.setState({list: list.slice(0, list.length - 1)});    // this will have race condition
        };
    
    render() {
        const {list} = this.state;
        
        return (
            <div>
            {
                list.map(i =>
                    <div key={i}>
                        {i}
                    </div>
                )
            }
            <button onClick={this.handleAdd}>Add</button>
            <button onClick={this.handleDrop}>Drop</button>
            </div>
        )
    }
}
```
As stated in comment, the above implementation will raise a race condition on list. As `this.setState` is async, the time 
of the call of `this.setState`, the passed in list may be different as the time of `this.setState`'s execution.  
```javascript
    handleAdd = () => {
        const {list} = this.state;
        this.setState({list: [...list, list[list.length - 1] + 1]});    // list is not synchronous update
    };
```
Fortunately, we can pass in a function that the argument `state` will be assigned the value of `state` at execution time.
```javascript
    handleAdd = () => {
        this.setState(state => (
            {...state, list: [...state.list, state.list[state.list.length - 1] + 1]})
        );
    };
```
Similarly we can modify `handleDrop` method.
```javascript
import React from "react";

class Group extends React.PureComponent {
    state = {
        list: [1, 2, 3, 4, 5]
    };
    
    handleAdd = () => {
        this.setState(state => (
            {...state, list: [...state.list, state.list[state.list.length - 1] + 1]}
        ));
    };
    
    handleDrop = () => {
        this.setState(state => (
            {...state, list: state.list.slice(0, state.list.length - 1)}
        ));
    };
    
    render() {
        const {list} = this.state;
        
        return (
            <div>
            {
                list.map(i =>
                    <div key={i}>
                        {i}
                    </div>
                )
            }
            <button onClick={this.handleAdd}>Add</button>
            <button onClick={this.handleDrop}>Drop</button>
            </div>
        )
    }
}
```
Note that this update is need when we call `setState` on the same state with in very short time. e.g.:
```javascript
this.setState({list: [...this.state.list, 99]});
this.setState({list: [...this.state.list, 100]});
```
In this case, the first call will be useless. When only one call on the same state is fired by a user event handle (e.g. 
onClick), the care for race condition is not necessary, as human operating is considerably slow.

We can see the newly-added list node appear or disappear abruptly. This is due to the node is inserted or removed based
on the change of `this.state.list` state. We can solve node abrupt appear easily by add css to the node. In Hysia-ServerFront,
we use third library transition [material-ui collapse](https://material-ui.com/utils/transitions/).
```javascript
// wrapped list node with Collapse
import React from "react";
import Collapse from "@material-ui/core/Collapse";

class Group extends React.PureComponent {
    
    state = {
        list: []
    };
    // ...
    
    render() {
        const {list} = this.state;
        
        return (
            <div>
            {
                list.map(i => 
                    <Collapse key={i} in={true}>
                        <div>
                            {i}
                        </div>
                    </Collapse>)
            }
            </div>
    )}
}
```
Problem arise when the list node dom is removed from the web page. Css cannot solve the problem as the dom is updated in 
a "hard" manner. Hereby we propose an approach: delay `list` state update.  
The flow becomes
```
user click drop ---> transition ---> update list
```
In implementation with Mui, we use anther list to present node display/no-display, so that we can control the happen of 
transition.
```javascript
handleOnDrop = () => {
    this.setState(state => {
        const listOn = state.listOn.slice();
        listOn[listOn.length - 1] = false;
        return ({...state, listOn: groupOn});
    })
};

handleDrop = () => {
    // to remove the unused list on flag
    const handleDropped = () => {
        this.setState(state => ({listOn: state.listOn.filter(x => x)}));
    };
    
    this.setState(state => (
        {list: state.list.slice(0, state.list.length - 1)}
    ), handleDropped);    // call back after drop is done
};
```
When user click drop, we first fire `handleOnDrop` to fire transition. When the transition finishes, `Collapse` component 
will fire a `exited` event, which we can supply a event listener `handleDrop`.
```
                                           extied
onClick ---> handleOnDrop ---> transition --------> handleDrop
```
```javascript
<div>
{
    list.map((i, index) =>
        <Collapse in={this.state.listOn[index]} onExited={this.handleDrop}>
            {/* some node */}
        </Collapse>
    )
}
    <button onClick={this.handleOnDrop}>Drop</button>
</div>
```
In this case, `Collapse` does not always have `true` assign to attribute `in`. We need to modify list add node logic.
```javascript
handleAdd = () => {
    // update list on flag
    const handleAdded = () => {
        this.setState(state => ({listOn: [...state.listOn, true]}));
    };

    this.setState(state => (
        {...state, list: [...state.list, state.list[state.list.length - 1] + 1]}
    ), handleAdded);    // call back after add is done
};
```
Of cause, we shall add dynamic initializer to `listOn`:
```javascript
componentDidMount() {
    this.setState({listOn: this.state.list.map(() => true)});
}
```
The full script is below
```javascript
import React from "react";
import Collapse from "@material-ui/core/Collapse";

class Group extends React.PureComponent {
    
    state = {
        list: [],
        listOn: [],  // the corresponding node true for the node will display
    };
    
    componentDidMount() {
        this.setState({listOn: this.state.list.map(() => true)});
    }
    
    handleOnDrop = () => {
        this.setState(state => {
            const listOn = state.listOn.slice();
            listOn[listOn.length - 1] = false;
            return ({...state, listOn: groupOn});
        })
    };
    
    handleDrop = () => {
        // to remove the unused list on flag
        const handleDropped = () => {
            this.setState(state => ({listOn: state.listOn.filter(x => x)}));
        };
        
        this.setState(state => (
            {list: state.list.slice(0, state.list.length - 1)}
        ), handleDropped);    // call back after drop is done
    };
    
    handleAdd = () => {
        // update list on flag
        const handleAdded = () => {
            this.setState(state => ({listOn: [...state.listOn, true]}));
        };
    
        this.setState(state => (
            {...state, list: [...state.list, state.list[state.list.length - 1] + 1]}
        ), handleAdded);    // call back after add is done
    };
    
    render() {
        const {list, listOn} = this.state;
        
        return (
            <div>
            {
                list.map((i, index) => 
                    <Collapse key={index} in={listOn[index]} onExited={this.handleDrop}>
                        <div>
                            {i}
                        </div>
                    </Collapse>)
            }
                <button onClick={this.handleAdd}>Add</button>
                <button onClick={this.handleOnDrop}>Drop</button>
            </div>
    )}
}
```
