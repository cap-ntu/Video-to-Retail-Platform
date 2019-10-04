import React from "react";

class MountTransition extends React.PureComponent {

    state = {
        mount: false,
        in: false,
    };

    handleIn = () => {
        this.setState({mount: true, in: true})
    };

    handleOnExit = () => {
        this.setState({in: false}, this.props.onExited)
    };

    handleOnExited = () => {
        this.setState({mount: false})
    };

    componentWillMount() {
        this.props.in ? this.handleIn() : this.handleOnExit();
    }

    componentWillUpdate(nextProps, nextState, nextContext) {
        if (nextProps.in !== this.props.in) {
            nextProps.in ? this.handleIn() : this.handleOnExit();
        }
    }

    render() {
        const {transition: Transition, children, ...rest} = this.props;
        const {mount, in: _in} = this.state;

        if (mount)
            return (
                <Transition in={_in} {...rest} onExited={this.handleOnExited}>
                    <div>
                        {children}
                    </div>
                </Transition>
            );
        return null;
    }
}

export default MountTransition;
