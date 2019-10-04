import React from 'react';
import PropTypes from 'prop-types';
import Typography from "@material-ui/core/Typography/Typography";
import {withStyles} from "@material-ui/core/styles";
import classNames from "classnames";
import ReactLink from "react-router-dom/Link";

const styles = {
    aUnderline: {
        '&:hover': {
            textDecoration: 'underline',
        }
    },
    aResponsive: {
        opacity: 0.7,
        transition: [['opacity', '0.25s', 'linear', '0.1s']],
        '&:hover': {
            opacity: 1,
        }
    },
};

const Link2 = ({
                   isRouter, classes, className, children, to = '/', variant = 'h6', color = 'inherit',
                  paragraph = false, noWrap = false, animation = true
              }) => (
    <div className={className}>
        <Typography className={classNames({
            [classes.aUnderline]: paragraph,
            [classes.aResponsive]: !paragraph && animation
        })
        }
                    component={props => (isRouter && !/http[s]?:\/\//.test(to)) ? <ReactLink to={to} {...props}/> :
                        <a href={to} {...props}/>}
                    variant={variant}
                    color={color}
                    noWrap={noWrap}>
            {children}
        </Typography>
    </div>
);

const A2 = props => <Link2 {...props} isRouter={false}/>;
const Link = props => <Link2 {...props} isRouter={true}/>;

A2.propTypes = Link.propTyes = {
    classes: PropTypes.object.isRequired,
    children: PropTypes.node,
    to: PropTypes.string,
    variant: PropTypes.oneOf([
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'subtitle1', 'subtitle2', 'body1', 'body2', 'caption', 'button',
        'overline', 'srOnly', 'inherit', "display4", 'display3', 'display2', 'display1', 'headline', 'title',
        'subheading',
    ]),
    color: PropTypes.oneOf(['default', 'error', 'inherit', 'primary', 'secondary', 'textPrimary', 'textSecondary',]),
    paragraph: PropTypes.bool,
    noWrap: PropTypes.bool,
};

export const A = withStyles(styles)(A2);
export default withStyles(styles)(Link);
