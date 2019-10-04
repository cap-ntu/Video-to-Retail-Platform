import React, {Component} from "react";
import * as PropTypes from "prop-types";
import Chip from "@material-ui/core/Chip";
import TextField from "@material-ui/core/TextField";
import withStyles from "@material-ui/core/styles/withStyles";
import IconButton from "@material-ui/core/es/IconButton/IconButton";
import AddCircleIcon from "@material-ui/icons/AddCircleRounded";
import Grow from "@material-ui/core/es/Grow/Grow";
import classNames from "classnames";

const styles = theme => ({
    chip: {
        margin: 0.5 * theme.spacing.unit,
    },
    textField: {
        transition: theme.transitions.create("width", {duration: theme.transitions.duration.short}),
        width: 200,
    },
    textFieldBlur: {
        width: 0,
    },
});

class ChipInput extends Component {
    state = {
        text: "",
        values: [],
        blur: true,
    };

    componentDidMount() {
        this.setState({values: this.props.values})
    }

    handleDelete = index => () => {
        this.props.values.splice(index, 1);
        this.setState({values: this.props.values});
    };

    handleChange = e => {
        this.setState({text: e.target.value});
    };

    handleKeyDown = e => {
        const {text} = this.state;
        if (e.keyCode === 13 && text !== "") {
            this.props.values.push(text);
            this.setState({text: "", values: this.props.values});
        }
    };

    render() {
        const {classes, id, label, placeholder} = this.props;
        const {text, values, blur} = this.state;
        return (
            <React.Fragment>{
                values.map((value, index) =>
                    <Grow key={index} in={true}>
                        <Chip className={classes.chip}
                              label={value}
                              onDelete={this.handleDelete(index)}/>
                    </Grow>)}
                <TextField
                    className={classNames(classes.textField, {[classes.textFieldBlur]: blur})}
                    id={id}
                    label={label}
                    placeholder={placeholder}
                    value={text}
                    onChange={this.handleChange}
                    onBlur={() => this.setState({blur: true})}
                    onKeyDown={this.handleKeyDown}
                    margin="normal"
                />
                <Grow in={blur}>
                    <IconButton onClick={() => this.setState({blur: false})}>
                        <AddCircleIcon/>
                    </IconButton>
                </Grow>
            </React.Fragment>
        );
    }
}

ChipInput.propTypes = {
    classes: PropTypes.object.isRequired,
    id: PropTypes.string,
    label: PropTypes.string,
    placeholder: PropTypes.string,
    values: PropTypes.arrayOf(
        PropTypes.string.isRequired,
    ).isRequired,
};

export default withStyles(styles)(ChipInput);
